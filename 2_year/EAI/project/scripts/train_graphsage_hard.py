"""Retrains GraphSAGE with matched hard (2-hop) negatives and a pairwise
BPR ranking loss (softplus(neg-pos)) instead of random negatives + BCE.
--decoder-batch-size chunks the decoder's pos/neg scoring calls the same
way train_graphsage.py's batched_decode does."""
import argparse
import os
import random
import time

import torch
import torch.nn.functional as F
from torch_geometric.transforms import RandomLinkSplit

from common.hard_negative_eval import cap_pairs, evaluate_hard
from common.training_loop import EarlyStopper, build_out_path, run_multi_seed, seed_run, write_epoch_csv, write_eval_csv
from data_utils import igraph_to_pyg_data, load_split_graphs
from hard_negatives import precompute_two_hop_targets, sample_matched_hard_negatives
from train_graphsage import GraphSAGELinkPredictor

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "GNNs", "results")


def run(args, tag: str = "", out_dir: str = None) -> dict:
    """Runs one full train+eval pass for args.seed."""
    seed_run(args.seed)
    rng = random.Random(args.seed)
    device = torch.device(args.device)

    out_path = build_out_path(args.out, tag, out_dir)

    print(f"Loading split graphs (train/dev/test)... (seed={args.seed})")
    graphs = load_split_graphs()
    data = {name: igraph_to_pyg_data(g) for name, g in graphs.items()}

    print("Precomputing 2-hop hard-negative targets (train/dev/test)...")
    two_hop = {name: precompute_two_hop_targets(g, cap_per_node=args.cap_per_node, seed=args.seed)
               for name, g in graphs.items()}
    for name, g in graphs.items():
        print(f"  {name}: {len(two_hop[name])}/{g.vcount()} nodes have >=1 hard-negative candidate")

    splitter = RandomLinkSplit(num_val=args.val_frac, num_test=args.test_frac, is_undirected=False,
                                add_negative_train_samples=False, neg_sampling_ratio=0.0)
    # num_val=num_test=0.0: whole graph used for message passing + eval
    full_splitter = RandomLinkSplit(num_val=0.0, num_test=0.0, is_undirected=False,
                                     add_negative_train_samples=False, neg_sampling_ratio=0.0)
    train_split, _, train_test_split = splitter(data["train"])
    dev_split, _, _ = full_splitter(data["dev"])
    test_split, _, _ = full_splitter(data["test"])

    train_split.edge_label_index = cap_pairs(train_split.edge_label_index, args.max_train_pairs, rng)
    train_test_split.edge_label_index = cap_pairs(train_test_split.edge_label_index, args.max_eval_pairs, rng)
    dev_split.edge_label_index = cap_pairs(dev_split.edge_label_index, args.max_eval_pairs, rng)
    test_split.edge_label_index = cap_pairs(test_split.edge_label_index, args.max_eval_pairs, rng)
    print(f"train/dev/test positive pairs: {train_split.edge_label_index.size(1)}/"
          f"{dev_split.edge_label_index.size(1)}/{test_split.edge_label_index.size(1)}")

    in_dim = data["train"].x.size(-1)
    model = GraphSAGELinkPredictor(in_dim, args.hidden_dim, args.num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_split = train_split.to(device)
    pos_pairs_train = train_split.edge_label_index
    train_sources = pos_pairs_train[0].tolist()
    num_nodes_train = data["train"].num_nodes

    os.makedirs(RESULTS_DIR, exist_ok=True)
    val_metric_name = args.val_metric
    epoch_rows = []  # (epoch, train_loss, dev_loss, dev_auc, dev_ap, elapsed_s)

    print(f"Training for up to {args.epochs} epochs "
          f"(model selection + early stopping on dev_{val_metric_name}, patience={args.patience})...")
    t0 = time.perf_counter()
    stopper = EarlyStopper(args.patience, args.min_delta)
    n_train_pairs = pos_pairs_train.size(1)
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()

        h = model.encoder(train_split.x, train_split.edge_index)

        neg_targets, n_hard, n_fallback = sample_matched_hard_negatives(
            train_sources, two_hop["train"], num_nodes_train, rng
        )
        neg_pairs_train = torch.tensor([train_sources, neg_targets], dtype=torch.long, device=device)

        # Per-chunk losses (reduction="sum" / n_train_pairs) accumulate to
        # the same gradient as one un-chunked backward() would give.
        # retain_graph keeps the shared encoder graph alive across chunks.
        total_loss = 0.0
        bs = args.decoder_batch_size or n_train_pairs
        for start in range(0, n_train_pairs, bs):
            end = min(start + bs, n_train_pairs)
            pos_chunk = model.predictor(h, pos_pairs_train[:, start:end])
            neg_chunk = model.predictor(h, neg_pairs_train[:, start:end])
            loss_chunk = F.softplus(neg_chunk - pos_chunk).sum()
            (loss_chunk / n_train_pairs).backward(retain_graph=(end < n_train_pairs))
            total_loss += loss_chunk.item()
        loss_value = total_loss / n_train_pairs
        optimizer.step()

        dev_auc, dev_ap, dev_loss, dev_hard, dev_fallback = evaluate_hard(
            model, dev_split, two_hop["dev"], data["dev"].num_nodes, rng, device, args.decoder_batch_size
        )
        dev_metric = dev_auc if val_metric_name == "auc" else dev_ap
        elapsed = time.perf_counter() - t0
        print(f"epoch {epoch:03d}  train_loss={loss_value:.4f}  dev_loss={dev_loss:.4f}  dev_auc={dev_auc:.4f}  "
              f"dev_ap={dev_ap:.4f}  (train hard-neg hit rate {n_hard}/{n_hard + n_fallback})  ({elapsed:.1f}s elapsed)")
        epoch_rows.append((epoch, loss_value, dev_loss, dev_auc, dev_ap, elapsed))

        if stopper.update(dev_metric):
            torch.save(model.state_dict(), out_path)
        elif stopper.should_stop():
            print(f"early stopping: dev_{val_metric_name} hasn't improved by >= "
                  f"{args.min_delta} for {args.patience} epochs (best={stopper.best_metric:.4f})")
            break
    best_dev_metric = stopper.best_metric

    epoch_csv = os.path.join(out_dir or RESULTS_DIR, f"graphsage_hard_epoch_metrics{tag}.csv")
    write_epoch_csv(epoch_csv, epoch_rows)
    print(f"per-epoch metrics written to {epoch_csv}")

    print(f"\nBest dev {val_metric_name} (hard negatives): {best_dev_metric:.4f} (checkpoint saved to {out_path})")
    model.load_state_dict(torch.load(out_path, map_location=device))

    train_auc, train_ap, *_ = evaluate_hard(model, train_test_split, two_hop["train"], num_nodes_train, rng, device, args.decoder_batch_size)
    dev_auc, dev_ap, *_ = evaluate_hard(model, dev_split, two_hop["dev"], data["dev"].num_nodes, rng, device, args.decoder_batch_size)
    test_auc, test_ap, *_ = evaluate_hard(model, test_split, two_hop["test"], data["test"].num_nodes, rng, device, args.decoder_batch_size)
    print("\nHeld-out link-existence performance (hard 2-hop negatives):")
    print(f"  train graph (transductive): AUC={train_auc:.4f}  AP={train_ap:.4f}")
    print(f"  dev graph   (inductive):    AUC={dev_auc:.4f}  AP={dev_ap:.4f}")
    print(f"  test graph  (inductive):    AUC={test_auc:.4f}  AP={test_ap:.4f}")

    eval_csv = os.path.join(out_dir or RESULTS_DIR, f"graphsage_hard_eval_metrics{tag}.csv")
    write_eval_csv(eval_csv, [
        ("train", "transductive", train_auc, train_ap),
        ("dev", "inductive", dev_auc, dev_ap),
        ("test", "inductive", test_auc, test_ap),
    ])
    print(f"held-out eval metrics written to {eval_csv}")

    return {
        "seed": args.seed,
        "train_auc": train_auc, "train_ap": train_ap,
        "dev_auc": dev_auc, "dev_ap": dev_ap,
        "test_auc": test_auc, "test_ap": test_ap,
    }


def main():
    parser = argparse.ArgumentParser(description="Retrain GraphSAGE with hard 2-hop negatives + a BPR ranking loss")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--val-frac", type=float, default=0.05)
    parser.add_argument("--test-frac", type=float, default=0.10)
    parser.add_argument("--cap-per-node", type=int, default=50,
                         help="cap on 2-hop candidates stored per anchor node")
    parser.add_argument("--max-train-pairs", type=int, default=None,
                         help="cap on number of positive training edges, e.g. for a budget-matched "
                              "comparison against train_seal_hard.py's --max-train-pairs; "
                              "default trains on every train-graph edge")
    parser.add_argument("--max-eval-pairs", type=int, default=None,
                         help="cap on number of positive train/dev/test edges evaluated; "
                              "default evaluates on every edge in each split")
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
                              "are written to GNNs/results/graphsage_hard_eval_seed.csv, one row per seed")
    parser.add_argument("--seed-dir", type=str, default=None,
                         help="with --seeds, write all per-seed checkpoints/epoch-metrics and the final "
                              "graphsage_hard_eval_seed.csv into this folder (created if needed) instead "
                              "of scattering them across the checkpoint's own directory and GNNs/results")
    parser.add_argument("--out", type=str, default="graphsage_link_predictor_hardneg.pt")
    parser.add_argument("--decoder-batch-size", type=int, default=None,
                         help="cap on how many candidate pairs the decoder scores in one forward/backward "
                              "pass; only useful to lower on CUDA OOM -- default (None) never chunks")
    args = parser.parse_args()

    if args.seeds:
        run_multi_seed(
            args, run, "graphsage_hard_eval_seed.csv",
            ["seed", "train_auc", "train_ap", "dev_auc", "dev_ap", "test_auc", "test_ap"],
            RESULTS_DIR,
        )
    else:
        run(args)


if __name__ == "__main__":
    main()
