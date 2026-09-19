"""Retrains SEAL (see train_seal.py) with matched hard (2-hop) negatives
per anchor and a pairwise BPR ranking loss instead of random negatives +
BCE."""
import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.data import Batch, Data
from torch_geometric.data import Dataset as TorchDataset

from common.training_loop import EarlyStopper, build_out_path, run_multi_seed, seed_run, write_epoch_csv, write_eval_csv
from data_utils import build_node_features, load_split_graphs
from hard_negatives import precompute_two_hop_targets, sample_matched_hard_negatives
from train_seal import MAX_DRNL_LABEL, SEALLinkPredictor, extract_enclosing_subgraph

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "GNNs", "results")


class SEALHardPairDataset(TorchDataset):
    """Matched (pos, hard-negative) pairs sharing anchor u. Hard-negative
    targets are precomputed once in __init__; subgraph extraction is
    deferred to get_pos/get_neg. Single-process only (shared self.rng)."""

    def __init__(self, g, node_features, two_hop_targets, num_hops, max_nodes, max_pairs, seed):
        super().__init__()
        self.g = g
        self.node_features = node_features
        self.num_hops = num_hops
        self.max_nodes = max_nodes
        self.rng = random.Random(seed)

        edge_pairs = list(set(g.get_edgelist()))
        self.rng.shuffle(edge_pairs)
        pos_pairs = edge_pairs[:max_pairs]
        sources = [u for u, _ in pos_pairs]

        neg_targets, self.n_hard, self.n_fallback = sample_matched_hard_negatives(
            sources, two_hop_targets, g.vcount(), self.rng
        )
        self.pos_pairs = pos_pairs
        self.neg_targets = neg_targets

    def len(self):
        return len(self.pos_pairs)

    def _extract(self, u, v):
        edge_index, node_labels, sub_x = extract_enclosing_subgraph(
            self.g, u, v, self.node_features, self.num_hops, self.max_nodes, self.rng
        )
        return Data(
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            node_label=torch.tensor(node_labels, dtype=torch.long),
            x=torch.tensor(sub_x, dtype=torch.float),
            num_nodes=len(node_labels),
        )

    def get_pos(self, idx):
        u, v = self.pos_pairs[idx]
        return self._extract(u, v)

    def get_neg(self, idx):
        u, _ = self.pos_pairs[idx]
        v_neg = self.neg_targets[idx]
        return self._extract(u, v_neg)

    def get(self, idx):
        # Required by the Dataset ABC; unused (see iterate_paired_batches)
        return self.get_pos(idx)


def iterate_paired_batches(dataset, batch_size, rng):
    """Yields (pos_batch, neg_batch) matched pairs, index order shuffled by rng."""
    idx = list(range(len(dataset)))
    rng.shuffle(idx)
    for i in range(0, len(idx), batch_size):
        chunk = idx[i:i + batch_size]
        pos_batch = Batch.from_data_list([dataset.get_pos(j) for j in chunk])
        neg_batch = Batch.from_data_list([dataset.get_neg(j) for j in chunk])
        yield pos_batch, neg_batch


@torch.no_grad()
def evaluate_pairs(model, dataset, batch_size, device):
    """AUC/AP over pooled pos/neg predictions, plus the paired BPR loss."""
    model.eval()
    all_probs, all_y = [], []
    total_loss, n_pairs = 0.0, 0
    n = len(dataset)
    for i in range(0, n, batch_size):
        chunk = range(i, min(i + batch_size, n))
        pos_chunk = [dataset.get_pos(j) for j in chunk]
        neg_chunk = [dataset.get_neg(j) for j in chunk]
        if not pos_chunk:
            continue
        pos_batch = Batch.from_data_list(pos_chunk).to(device)
        neg_batch = Batch.from_data_list(neg_chunk).to(device)
        pos_score = model(pos_batch)
        neg_score = model(neg_batch)
        total_loss += F.softplus(neg_score - pos_score).sum().item()
        n_pairs += len(pos_chunk)
        for logits, label in ((pos_score, 1.0), (neg_score, 0.0)):
            all_probs.append(torch.sigmoid(logits).cpu().numpy())
            all_y.append(np.full(logits.size(0), label))
    probs = np.concatenate(all_probs)
    y = np.concatenate(all_y)
    avg_loss = total_loss / max(n_pairs, 1)
    return roc_auc_score(y, probs), average_precision_score(y, probs), avg_loss


def run(args, tag: str = "", out_dir: str = None) -> dict:
    """Runs one full train+eval pass for args.seed."""
    seed_run(args.seed)
    rng = random.Random(args.seed)
    device = torch.device(args.device)

    out_path = build_out_path(args.out, tag, out_dir)

    print(f"Loading split graphs (train/dev/test)... (seed={args.seed})")
    graphs = load_split_graphs()
    node_features = {name: build_node_features(g) for name, g in graphs.items()}
    content_dim = next(iter(node_features.values())).shape[1]

    print("Precomputing 2-hop hard-negative targets (train/dev/test)...")
    two_hop = {name: precompute_two_hop_targets(g, cap_per_node=args.cap_per_node, seed=args.seed)
               for name, g in graphs.items()}

    print(f"Extracting enclosing subgraphs (num_hops={args.num_hops})...")
    train_set = SEALHardPairDataset(
        graphs["train"], node_features["train"], two_hop["train"],
        args.num_hops, args.max_nodes, args.max_train_pairs, args.seed
    )
    dev_set = SEALHardPairDataset(
        graphs["dev"], node_features["dev"], two_hop["dev"],
        args.num_hops, args.max_nodes, args.max_eval_pairs, args.seed
    )
    test_set = SEALHardPairDataset(
        graphs["test"], node_features["test"], two_hop["test"],
        args.num_hops, args.max_nodes, args.max_eval_pairs, args.seed
    )
    print(f"train/dev/test pairs: {len(train_set)}/{len(dev_set)}/{len(test_set)}  "
          f"(train hard-neg hit rate {train_set.n_hard}/{train_set.n_hard + train_set.n_fallback})")

    model = SEALLinkPredictor(MAX_DRNL_LABEL, args.label_dim, content_dim,
                               hidden_dim=args.hidden_dim, k=args.sort_k).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    val_metric_name = args.val_metric
    epoch_rows = []  # (epoch, train_loss, dev_loss, dev_auc, dev_ap, elapsed_s)

    print(f"Training for up to {args.epochs} epochs "
          f"(model selection + early stopping on dev_{val_metric_name}, patience={args.patience})...")
    t0 = time.perf_counter()
    stopper = EarlyStopper(args.patience, args.min_delta)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, n = 0.0, 0
        for pos_batch, neg_batch in iterate_paired_batches(train_set, args.batch_size, rng):
            pos_batch, neg_batch = pos_batch.to(device), neg_batch.to(device)
            optimizer.zero_grad()
            pos_score = model(pos_batch)
            neg_score = model(neg_batch)
            loss = F.softplus(neg_score - pos_score).mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * pos_batch.num_graphs
            n += pos_batch.num_graphs

        avg_loss = total_loss / max(n, 1)
        dev_auc, dev_ap, dev_loss = evaluate_pairs(model, dev_set, args.batch_size, device)
        dev_metric = dev_auc if val_metric_name == "auc" else dev_ap
        elapsed = time.perf_counter() - t0
        print(f"epoch {epoch:03d}  train_loss={avg_loss:.4f}  dev_loss={dev_loss:.4f}  "
              f"dev_auc={dev_auc:.4f}  dev_ap={dev_ap:.4f}  ({elapsed:.1f}s elapsed)")
        epoch_rows.append((epoch, avg_loss, dev_loss, dev_auc, dev_ap, elapsed))

        if stopper.update(dev_metric):
            torch.save(model.state_dict(), out_path)
        elif stopper.should_stop():
            print(f"early stopping: dev_{val_metric_name} hasn't improved by >= "
                  f"{args.min_delta} for {args.patience} epochs (best={stopper.best_metric:.4f})")
            break
    best_dev_metric = stopper.best_metric

    epoch_csv = os.path.join(out_dir or RESULTS_DIR, f"seal_hard_epoch_metrics{tag}.csv")
    write_epoch_csv(epoch_csv, epoch_rows)
    print(f"per-epoch metrics written to {epoch_csv}")

    print(f"\nBest dev {val_metric_name} (hard negatives): {best_dev_metric:.4f} (checkpoint saved to {out_path})")
    model.load_state_dict(torch.load(out_path, map_location=device))
    dev_auc, dev_ap, _ = evaluate_pairs(model, dev_set, args.batch_size, device)
    test_auc, test_ap, test_loss = evaluate_pairs(model, test_set, args.batch_size, device)
    print(f"\nTest loss: {test_loss:.4f}  Test AUC (hard negatives): {test_auc:.4f}  Test AP: {test_ap:.4f}")

    eval_csv = os.path.join(out_dir or RESULTS_DIR, f"seal_hard_eval_metrics{tag}.csv")
    write_eval_csv(eval_csv, [
        ("dev", "inductive (model selection)", dev_auc, dev_ap),
        ("test", "inductive", test_auc, test_ap),
    ])
    print(f"held-out eval metrics written to {eval_csv}")

    return {"seed": args.seed, "dev_auc": dev_auc, "dev_ap": dev_ap, "test_auc": test_auc, "test_ap": test_ap}


def main():
    parser = argparse.ArgumentParser(description="Retrain SEAL with hard 2-hop negatives + a BPR ranking loss")
    parser.add_argument("--num-hops", type=int, default=1)
    parser.add_argument("--max-nodes", type=int, default=100)
    parser.add_argument("--sort-k", type=int, default=30)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--label-dim", type=int, default=16)
    parser.add_argument("--cap-per-node", type=int, default=50,
                         help="cap on 2-hop candidates stored per anchor node")
    parser.add_argument("--max-train-pairs", type=int, default=20000)
    parser.add_argument("--max-eval-pairs", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
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
                              "are written to GNNs/results/seal_hard_eval_seed.csv, one row per seed")
    parser.add_argument("--seed-dir", type=str, default=None,
                         help="with --seeds, write all per-seed checkpoints/epoch-metrics and the final "
                              "seal_hard_eval_seed.csv into this folder (created if needed) instead of "
                              "scattering them across the checkpoint's own directory and GNNs/results")
    parser.add_argument("--out", type=str, default="seal_link_predictor_hardneg.pt")
    args = parser.parse_args()

    if args.seeds:
        run_multi_seed(
            args, run, "seal_hard_eval_seed.csv",
            ["seed", "dev_auc", "dev_ap", "test_auc", "test_ap"],
            RESULTS_DIR,
        )
    else:
        run(args)


if __name__ == "__main__":
    main()
