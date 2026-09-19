"""Hard-negative counterpart to mlp_baseline_common.py: same 2-hop
negatives and BPR loss as train_graphsage_hard.py, but with
MLPBaselineLinkPredictor (no message passing)."""
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
from mlp_baseline_common import MLPBaselineLinkPredictor

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "GNNs", "results")


def run(args, variant: str, tag: str = "", out_dir: str = None) -> dict:
    """Runs one full train+eval pass for args.seed."""
    seed_run(args.seed)
    rng = random.Random(args.seed)
    device = torch.device(args.device)

    out_path = build_out_path(args.out, tag, out_dir)

    print(f"Loading split graphs (train/dev/test)... (variant={variant}, seed={args.seed})")
    graphs = load_split_graphs()
    data = {name: igraph_to_pyg_data(g) for name, g in graphs.items()}

    print("Precomputing 2-hop hard-negative targets (train/dev/test)...")
    two_hop = {name: precompute_two_hop_targets(g, cap_per_node=args.cap_per_node, seed=args.seed)
               for name, g in graphs.items()}
    for name, g in graphs.items():
        print(f"  {name}: {len(two_hop[name])}/{g.vcount()} nodes have >=1 hard-negative candidate")

    splitter = RandomLinkSplit(num_val=args.val_frac, num_test=args.test_frac, is_undirected=False,
                                add_negative_train_samples=False, neg_sampling_ratio=0.0)
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
    model = MLPBaselineLinkPredictor(in_dim, args.hidden_dim, args.num_layers, variant=variant).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_split = train_split.to(device)
    pos_pairs_train = train_split.edge_label_index
    train_sources = pos_pairs_train[0].tolist()
    num_nodes_train = data["train"].num_nodes

    os.makedirs(RESULTS_DIR, exist_ok=True)
    val_metric_name = args.val_metric
    epoch_rows = []

    print(f"Training for up to {args.epochs} epochs "
          f"(model selection + early stopping on dev_{val_metric_name}, patience={args.patience})...")
    t0 = time.perf_counter()
    stopper = EarlyStopper(args.patience, args.min_delta)
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()

        h = model.encoder(train_split.x) if model.encoder is not None else train_split.x
        pos_score = model.decoder(h, pos_pairs_train)

        neg_targets, n_hard, n_fallback = sample_matched_hard_negatives(
            train_sources, two_hop["train"], num_nodes_train, rng
        )
        neg_pairs = torch.tensor([train_sources, neg_targets], dtype=torch.long, device=device)
        neg_score = model.decoder(h, neg_pairs)

        loss = F.softplus(neg_score - pos_score).mean()
        loss.backward()
        optimizer.step()

        dev_auc, dev_ap, dev_loss, dev_hard, dev_fallback = evaluate_hard(
            model, dev_split, two_hop["dev"], data["dev"].num_nodes, rng, device
        )
        dev_metric = dev_auc if val_metric_name == "auc" else dev_ap
        elapsed = time.perf_counter() - t0
        print(f"epoch {epoch:03d}  train_loss={loss.item():.4f}  dev_loss={dev_loss:.4f}  dev_auc={dev_auc:.4f}  "
              f"dev_ap={dev_ap:.4f}  (train hard-neg hit rate {n_hard}/{n_hard + n_fallback})  ({elapsed:.1f}s elapsed)")
        epoch_rows.append((epoch, loss.item(), dev_loss, dev_auc, dev_ap, elapsed))

        if stopper.update(dev_metric):
            torch.save(model.state_dict(), out_path)
        elif stopper.should_stop():
            print(f"early stopping: dev_{val_metric_name} hasn't improved by >= "
                  f"{args.min_delta} for {args.patience} epochs (best={stopper.best_metric:.4f})")
            break
    best_dev_metric = stopper.best_metric

    epoch_csv = os.path.join(out_dir or RESULTS_DIR, f"mlp_baseline_{variant}_hard_epoch_metrics{tag}.csv")
    write_epoch_csv(epoch_csv, epoch_rows)
    print(f"per-epoch metrics written to {epoch_csv}")

    print(f"\nBest dev {val_metric_name} (hard negatives): {best_dev_metric:.4f} (checkpoint saved to {out_path})")
    model.load_state_dict(torch.load(out_path, map_location=device))

    train_auc, train_ap, *_ = evaluate_hard(model, train_test_split, two_hop["train"], num_nodes_train, rng, device)
    dev_auc, dev_ap, *_ = evaluate_hard(model, dev_split, two_hop["dev"], data["dev"].num_nodes, rng, device)
    test_auc, test_ap, *_ = evaluate_hard(model, test_split, two_hop["test"], data["test"].num_nodes, rng, device)
    print(f"\nHeld-out link-existence performance (variant={variant}, hard 2-hop negatives):")
    print(f"  train graph (transductive): AUC={train_auc:.4f}  AP={train_ap:.4f}")
    print(f"  dev graph   (inductive):    AUC={dev_auc:.4f}  AP={dev_ap:.4f}")
    print(f"  test graph  (inductive):    AUC={test_auc:.4f}  AP={test_ap:.4f}")

    eval_csv = os.path.join(out_dir or RESULTS_DIR, f"mlp_baseline_{variant}_hard_eval_metrics{tag}.csv")
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


def build_arg_parser(variant: str, default_out: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f"Retrain the graph-free '{variant}' link-existence baseline with hard 2-hop "
                    f"negatives + a BPR ranking loss (no message passing)"
    )
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2,
                         help="depth of the per-node MLP (ignored for variant=raw, which has no encoder)")
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
    parser.add_argument("--val-metric", type=str, default="auc", choices=["auc", "ap"])
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--seed-dir", type=str, default=None)
    parser.add_argument("--out", type=str, default=default_out)
    return parser


def main_for_variant(variant: str, default_out: str) -> None:
    args = build_arg_parser(variant, default_out).parse_args()

    if args.seeds:
        run_multi_seed(
            args,
            lambda a, tag, out_dir: run(a, variant, tag=tag, out_dir=out_dir),
            f"mlp_baseline_{variant}_hard_eval_seed.csv",
            ["seed", "train_auc", "train_ap", "dev_auc", "dev_ap", "test_auc", "test_ap"],
            RESULTS_DIR,
        )
    else:
        run(args, variant)
