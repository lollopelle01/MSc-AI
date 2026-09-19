"""Trains a GraphSAGE link-existence predictor on the directed citation
graph. Encoder does full-batch message passing over observed edges;
decoder scores pairs via an asymmetric MLP on [h_u || h_v] (a dot product
can't distinguish u->v from v->u). --decoder-batch-size chunks only the
decoder's scoring step for memory-constrained GPUs; see batched_decode()."""
import argparse
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.nn import SAGEConv
from torch_geometric.transforms import RandomLinkSplit

from common.training_loop import EarlyStopper, _phase, build_out_path, run_multi_seed, seed_run, write_epoch_csv, write_eval_csv
from data_utils import igraph_to_pyg_data, load_split_graphs
from mlp_baseline_common import PairMLPDecoder

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "GNNs", "results")


class SAGEEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.convs = nn.ModuleList([SAGEConv(in_dim, hidden_dim)])
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class GraphSAGELinkPredictor(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 128, num_layers: int = 2):
        super().__init__()
        self.encoder = SAGEEncoder(in_dim, hidden_dim, num_layers)
        # self.predictor (not self.decoder): train_graphsage_hard.py calls
        # model.predictor(h, edge_label_index) directly, bypassing forward()
        self.predictor = PairMLPDecoder(hidden_dim, hidden_dim)

    def forward(self, x, edge_index, edge_label_index):
        h = self.encoder(x, edge_index)
        return self.predictor(h, edge_label_index)


def batched_decode(model, h, edge_label_index, batch_size):
    """Scores edge_label_index through model.predictor in chunks of
    batch_size pairs, concatenated back in order. No-op wrapper when
    batch_size is None or >= the pair count."""
    n = edge_label_index.size(1)
    if batch_size is None or batch_size >= n:
        return model.predictor(h, edge_label_index)
    return torch.cat([
        model.predictor(h, edge_label_index[:, start:start + batch_size])
        for start in range(0, n, batch_size)
    ], dim=0)


@torch.no_grad()
def evaluate(model, split_data, device, decoder_batch_size=None):
    model.eval()
    split_data = split_data.to(device)
    h = model.encoder(split_data.x, split_data.edge_index)
    logits = batched_decode(model, h, split_data.edge_label_index, decoder_batch_size)
    loss = F.binary_cross_entropy_with_logits(logits, split_data.edge_label).item()
    probs = torch.sigmoid(logits).cpu().numpy()
    y = split_data.edge_label.cpu().numpy()
    auc = roc_auc_score(y, probs)
    ap = average_precision_score(y, probs)
    return auc, ap, loss


def run(args, tag: str = "", out_dir: str = None) -> dict:
    """Runs one full train+eval pass for args.seed."""
    seed_run(args.seed)
    device = torch.device(args.device)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    run_start = time.perf_counter()

    out_path = build_out_path(args.out, tag, out_dir)

    print(f"[phase] loading split graphs (train/dev/test)... (seed={args.seed})")
    t0 = time.perf_counter()
    graphs = load_split_graphs()
    _phase("loading split graphs", t0)

    print("[phase] building PyG data + RandomLinkSplit...")
    t0 = time.perf_counter()
    data = {name: igraph_to_pyg_data(g) for name, g in graphs.items()}

    splitter = RandomLinkSplit(
        num_val=args.val_frac,
        num_test=args.test_frac,
        is_undirected=False,
        add_negative_train_samples=True,
        neg_sampling_ratio=args.neg_sampling_ratio,
    )
    # num_val=num_test=0.0: whole graph used for message passing + eval
    full_splitter = RandomLinkSplit(
        num_val=0.0, num_test=0.0, is_undirected=False,
        add_negative_train_samples=True, neg_sampling_ratio=args.neg_sampling_ratio,
    )

    train_split, _, train_test_split = splitter(data["train"])
    dev_split, _, _ = full_splitter(data["dev"])
    test_split, _, _ = full_splitter(data["test"])
    _phase("building PyG data + RandomLinkSplit", t0)

    in_dim = data["train"].x.size(-1)
    model = GraphSAGELinkPredictor(in_dim, args.hidden_dim, args.num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_split = train_split.to(device)

    epoch_rows = []  # (epoch, train_loss, dev_loss, dev_auc, dev_ap, elapsed_s)
    ckpt_exists = os.path.exists(out_path)
    val_metric_name = args.val_metric

    if ckpt_exists:
        print(f"[phase] {out_path} already exists -- skipping training, "
              f"loading checkpoint for validation/test only")
        model.load_state_dict(torch.load(out_path, map_location=device))
        dev_auc, dev_ap, _ = evaluate(model, dev_split, device, args.decoder_batch_size)
        best_dev_metric = dev_auc if val_metric_name == "auc" else dev_ap
    else:
        print(f"[phase] training for up to {args.epochs} epochs "
              f"(model selection + early stopping on dev_{val_metric_name}, patience={args.patience})...")
        t0 = time.perf_counter()
        stopper = EarlyStopper(args.patience, args.min_delta)
        n_train_pairs = train_split.edge_label_index.size(1)
        for epoch in range(1, args.epochs + 1):
            model.train()
            optimizer.zero_grad()
            # Per-chunk losses (reduction="sum" / n_train_pairs) accumulate
            # to the same gradient as one un-chunked backward() would give.
            # retain_graph keeps the shared encoder graph alive across chunks.
            h = model.encoder(train_split.x, train_split.edge_index)
            total_loss = 0.0
            bs = args.decoder_batch_size or n_train_pairs
            for start in range(0, n_train_pairs, bs):
                end = min(start + bs, n_train_pairs)
                logits_chunk = model.predictor(h, train_split.edge_label_index[:, start:end])
                label_chunk = train_split.edge_label[start:end]
                loss_chunk = F.binary_cross_entropy_with_logits(logits_chunk, label_chunk, reduction="sum")
                (loss_chunk / n_train_pairs).backward(retain_graph=(end < n_train_pairs))
                total_loss += loss_chunk.item()
            loss_value = total_loss / n_train_pairs
            optimizer.step()

            dev_auc, dev_ap, dev_loss = evaluate(model, dev_split, device, args.decoder_batch_size)
            dev_metric = dev_auc if val_metric_name == "auc" else dev_ap
            elapsed = time.perf_counter() - t0
            print(f"epoch {epoch:03d}  train_loss={loss_value:.4f}  dev_loss={dev_loss:.4f}  "
                  f"dev_auc={dev_auc:.4f}  dev_ap={dev_ap:.4f}  ({elapsed:.1f}s elapsed)")
            epoch_rows.append((epoch, loss_value, dev_loss, dev_auc, dev_ap, elapsed))

            if stopper.update(dev_metric):
                torch.save(model.state_dict(), out_path)
            elif stopper.should_stop():
                print(f"[phase] early stopping: dev_{val_metric_name} hasn't improved by >= "
                      f"{args.min_delta} for {args.patience} epochs (best={stopper.best_metric:.4f})")
                break
        best_dev_metric = stopper.best_metric
        _phase(f"training ({len(epoch_rows)} epochs)", t0)

        epoch_csv = os.path.join(out_dir or RESULTS_DIR, f"graphsage_epoch_metrics{tag}.csv")
        write_epoch_csv(epoch_csv, epoch_rows)
        print(f"[phase] per-epoch metrics written to {epoch_csv}")

    print(f"\nBest dev {val_metric_name}: {best_dev_metric:.4f} (checkpoint: {out_path})")
    model.load_state_dict(torch.load(out_path, map_location=device))

    print("[phase] evaluating held-out train/dev/test...")
    t0 = time.perf_counter()
    train_auc, train_ap, _ = evaluate(model, train_test_split, device, args.decoder_batch_size)
    dev_auc, dev_ap, _ = evaluate(model, dev_split, device, args.decoder_batch_size)
    test_auc, test_ap, _ = evaluate(model, test_split, device, args.decoder_batch_size)
    _phase("evaluating held-out train/dev/test", t0)

    print("\nHeld-out link-existence performance:")
    print(f"  train graph (transductive): AUC={train_auc:.4f}  AP={train_ap:.4f}")
    print(f"  dev graph   (inductive):    AUC={dev_auc:.4f}  AP={dev_ap:.4f}")
    print(f"  test graph  (inductive):    AUC={test_auc:.4f}  AP={test_ap:.4f}")

    eval_csv = os.path.join(out_dir or RESULTS_DIR, f"graphsage_eval_metrics{tag}.csv")
    write_eval_csv(eval_csv, [
        ("train", "transductive", train_auc, train_ap),
        ("dev", "inductive", dev_auc, dev_ap),
        ("test", "inductive", test_auc, test_ap),
    ])
    print(f"[phase] held-out eval metrics written to {eval_csv}")
    _phase("total run time", run_start)

    return {
        "seed": args.seed,
        "train_auc": train_auc, "train_ap": train_ap,
        "dev_auc": dev_auc, "dev_ap": dev_ap,
        "test_auc": test_auc, "test_ap": test_ap,
    }


def main():
    parser = argparse.ArgumentParser(description="Train a GraphSAGE link-existence predictor on the SciCite citation graph")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--val-frac", type=float, default=0.05)
    parser.add_argument("--test-frac", type=float, default=0.10)
    parser.add_argument("--neg-sampling-ratio", type=float, default=1.0)
    parser.add_argument("--val-metric", type=str, default="auc", choices=["auc", "ap"],
                         help="validation metric used for checkpoint selection and early stopping")
    parser.add_argument("--patience", type=int, default=10,
                         help="stop early after this many epochs with no val-metric improvement; <=0 disables early stopping")
    parser.add_argument("--min-delta", type=float, default=1e-3,
                         help="minimum increase in val-metric to count as an improvement")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                         help="run the full train+eval pipeline once per seed (overrides --seed); "
                              "each run fixes torch/random/CUDA RNGs from its own seed, and results "
                              "are written to GNNs/results/graphsage_eval_seed.csv, one row per seed")
    parser.add_argument("--seed-dir", type=str, default=None,
                         help="with --seeds, write all per-seed checkpoints/epoch-metrics/eval-metrics "
                              "and the final graphsage_eval_seed.csv into this folder (created if needed) "
                              "instead of scattering them across the checkpoint's own directory and "
                              "GNNs/results")
    parser.add_argument("--out", type=str, default="graphsage_link_predictor.pt")
    parser.add_argument("--decoder-batch-size", type=int, default=None,
                         help="cap on how many candidate pairs the decoder scores in one forward/backward "
                              "pass; the encoder still runs full-graph, unbatched, exactly as always -- "
                              "only the decoder's per-pair MLP call is chunked (see batched_decode's "
                              "docstring for why this changes memory usage, not the training math). "
                              "Gradients across chunks are accumulated to match the un-chunked loss/gradient "
                              "exactly, so results are unaffected; only useful to lower this if a run hits "
                              "CUDA OOM (e.g. with a larger --hidden-dim) -- default (None) never chunks")
    args = parser.parse_args()

    if args.seeds:
        run_multi_seed(
            args, run, "graphsage_eval_seed.csv",
            ["seed", "train_auc", "train_ap", "dev_auc", "dev_ap", "test_auc", "test_ap"],
            RESULTS_DIR,
        )
    else:
        run(args)


if __name__ == "__main__":
    main()
