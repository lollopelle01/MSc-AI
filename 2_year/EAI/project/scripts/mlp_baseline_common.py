"""Shared training driver for graph-FREE link-existence baselines: same
node features as train_graphsage.py, but no message passing over
edge_index -- reference points for how much GraphSAGE's neighbourhood
aggregation buys over the same features/splits/decoder/eval protocol.

Two variants:
  "raw"     -- decoder MLP scores [x_u || x_v] directly from raw features,
              no per-node transform.
  "encoded" -- a per-node feed-forward MLP (same depth/width as
              train_graphsage.py's SAGEEncoder, plain nn.Linear instead of
              SAGEConv) transforms each node independently before the
              decoder scores [h_u || h_v].

    pip install torch torch_geometric scikit-learn
    python scripts/train_mlp_baseline_raw.py --epochs 50
    python scripts/train_mlp_baseline_encoded.py --epochs 50
"""
import argparse
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.transforms import RandomLinkSplit

from common.training_loop import EarlyStopper, _phase, build_out_path, run_multi_seed, seed_run, write_epoch_csv, write_eval_csv
from data_utils import igraph_to_pyg_data, load_split_graphs

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "GNNs", "results")


class NodeMLPEncoder(nn.Module):
    """Per-node feed-forward transform; never touches edge_index. Same
    depth/width as SAGEEncoder (train_graphsage.py), nn.Linear instead of
    SAGEConv."""

    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(in_dim, hidden_dim)])
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class PairMLPDecoder(nn.Module):
    """[h_u || h_v] -> MLP -> logit. in_dim is decoupled from hidden_dim so
    both the "raw" (in_dim = full node-feature width) and "encoded"
    (in_dim = hidden_dim) variants can share this class."""

    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h: torch.Tensor, edge_label_index: torch.Tensor) -> torch.Tensor:
        h_src = h[edge_label_index[0]]
        h_tgt = h[edge_label_index[1]]
        return self.mlp(torch.cat([h_src, h_tgt], dim=-1)).squeeze(-1)


class MLPBaselineLinkPredictor(nn.Module):
    """variant="raw": decoder scores raw node features directly.
    variant="encoded": a NodeMLPEncoder runs first. edge_index is accepted
    in forward() to match GraphSAGELinkPredictor's signature but never read."""

    def __init__(self, in_dim: int, hidden_dim: int = 128, num_layers: int = 2, variant: str = "raw"):
        super().__init__()
        if variant not in ("raw", "encoded"):
            raise ValueError(f"variant must be 'raw' or 'encoded', got {variant!r}")
        self.variant = variant
        if variant == "encoded":
            self.encoder = NodeMLPEncoder(in_dim, hidden_dim, num_layers)
            decoder_in_dim = hidden_dim
        else:
            self.encoder = None
            decoder_in_dim = in_dim
        self.decoder = PairMLPDecoder(decoder_in_dim, hidden_dim)

    def forward(self, x, edge_index, edge_label_index):
        h = self.encoder(x) if self.encoder is not None else x
        return self.decoder(h, edge_label_index)


def batched_decode(model, h, edge_label_index, batch_size):
    """Scores edge_label_index through model.decoder in chunks of
    `batch_size` pairs, concatenated back in order. Most relevant for
    variant="raw", where h is the full raw feature width on every pair.
    No-op wrapper when batch_size is None or >= the pair count."""
    n = edge_label_index.size(1)
    if batch_size is None or batch_size >= n:
        return model.decoder(h, edge_label_index)
    return torch.cat([
        model.decoder(h, edge_label_index[:, start:start + batch_size])
        for start in range(0, n, batch_size)
    ], dim=0)


@torch.no_grad()
def evaluate(model, split_data, device, decoder_batch_size=None):
    model.eval()
    split_data = split_data.to(device)
    h = model.encoder(split_data.x) if model.encoder is not None else split_data.x
    logits = batched_decode(model, h, split_data.edge_label_index, decoder_batch_size)
    loss = F.binary_cross_entropy_with_logits(logits, split_data.edge_label).item()
    probs = torch.sigmoid(logits).cpu().numpy()
    y = split_data.edge_label.cpu().numpy()
    auc = roc_auc_score(y, probs)
    ap = average_precision_score(y, probs)
    return auc, ap, loss


def run(args, variant: str, tag: str = "", out_dir: str = None) -> dict:
    """Runs one full train+eval pass for args.seed."""
    seed_run(args.seed)
    device = torch.device(args.device)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    run_start = time.perf_counter()

    out_path = build_out_path(args.out, tag, out_dir)

    print(f"[phase] loading split graphs (train/dev/test)... (variant={variant}, seed={args.seed})")
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
    full_splitter = RandomLinkSplit(
        num_val=0.0, num_test=0.0, is_undirected=False,
        add_negative_train_samples=True, neg_sampling_ratio=args.neg_sampling_ratio,
    )

    train_split, _, train_test_split = splitter(data["train"])
    dev_split, _, _ = full_splitter(data["dev"])
    test_split, _, _ = full_splitter(data["test"])
    _phase("building PyG data + RandomLinkSplit", t0)

    in_dim = data["train"].x.size(-1)
    model = MLPBaselineLinkPredictor(in_dim, args.hidden_dim, args.num_layers, variant=variant).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_split = train_split.to(device)

    epoch_rows = []
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
            # Per-chunk losses accumulate to the same gradient as one
            # un-chunked backward() would give (see train_graphsage.py).
            h = model.encoder(train_split.x) if model.encoder is not None else train_split.x
            total_loss = 0.0
            bs = args.decoder_batch_size or n_train_pairs
            for start in range(0, n_train_pairs, bs):
                end = min(start + bs, n_train_pairs)
                logits_chunk = model.decoder(h, train_split.edge_label_index[:, start:end])
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

        epoch_csv = os.path.join(out_dir or RESULTS_DIR, f"mlp_baseline_{variant}_epoch_metrics{tag}.csv")
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

    print(f"\nHeld-out link-existence performance (variant={variant}):")
    print(f"  train graph (transductive): AUC={train_auc:.4f}  AP={train_ap:.4f}")
    print(f"  dev graph   (inductive):    AUC={dev_auc:.4f}  AP={dev_ap:.4f}")
    print(f"  test graph  (inductive):    AUC={test_auc:.4f}  AP={test_ap:.4f}")

    eval_csv = os.path.join(out_dir or RESULTS_DIR, f"mlp_baseline_{variant}_eval_metrics{tag}.csv")
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


def build_arg_parser(variant: str, default_out: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f"Train the graph-free '{variant}' link-existence baseline on the SciCite citation graph "
                    f"(same features/splits/eval as train_graphsage.py, no message passing)"
    )
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2,
                         help="depth of the per-node MLP (ignored for variant=raw, which has no encoder)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--val-frac", type=float, default=0.05)
    parser.add_argument("--test-frac", type=float, default=0.10)
    parser.add_argument("--neg-sampling-ratio", type=float, default=1.0)
    parser.add_argument("--val-metric", type=str, default="auc", choices=["auc", "ap"])
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--seed-dir", type=str, default=None)
    parser.add_argument("--out", type=str, default=default_out)
    parser.add_argument("--decoder-batch-size", type=int, default=None,
                         help="cap on how many candidate pairs the decoder scores in one forward/backward "
                              "pass; only useful to lower on CUDA OOM -- default (None) never chunks")
    return parser


def main_for_variant(variant: str, default_out: str) -> None:
    args = build_arg_parser(variant, default_out).parse_args()

    if args.seeds:
        run_multi_seed(
            args,
            lambda a, tag, out_dir: run(a, variant, tag=tag, out_dir=out_dir),
            f"mlp_baseline_{variant}_eval_seed.csv",
            ["seed", "train_auc", "train_ap", "dev_auc", "dev_ap", "test_auc", "test_ap"],
            RESULTS_DIR,
        )
    else:
        run(args, variant)
