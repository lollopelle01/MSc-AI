"""Trains a SEAL link-existence predictor (Zhang & Chen, 2018) on the
directed citation graph: extracts each pair's enclosing subgraph, labels
nodes with DRNL, and scores it with a GNN."""
import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.data import Data
from torch_geometric.data import Dataset as TorchDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_sort_pool
from torch_geometric.utils import negative_sampling

from common.training_loop import EarlyStopper, _phase, build_out_path, run_multi_seed, seed_run, write_epoch_csv, write_eval_csv
from data_utils import build_node_features, load_split_graphs

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "GNNs", "results")

MAX_DRNL_LABEL = 20  # DRNL labels are capped/clipped at this value


def drnl_label(dist_u, dist_v, max_label=MAX_DRNL_LABEL):
    if dist_u is None or dist_v is None or np.isinf(dist_u) or np.isinf(dist_v):
        return 0  # w is not reachable from u, or can't reach v
    dist_u, dist_v = int(dist_u), int(dist_v)
    if dist_u == 0 or dist_v == 0:
        return 1  # u or v itself
    d = dist_u + dist_v
    label = 1 + min(dist_u, dist_v) + (d // 2) * (d // 2 + d % 2 - 1)
    return min(label, max_label)


def extract_enclosing_subgraph(g, u, v, node_features, num_hops=1, max_nodes=100, rng=None):
    """Returns (edge_index, drnl_labels, sub_x) for the enclosing subgraph
    of directed pair (u, v). node_features is the full split's feature
    matrix, indexed by igraph vertex id."""
    nb_u = set(g.neighborhood(vertices=u, order=num_hops, mode="all"))
    nb_v = set(g.neighborhood(vertices=v, order=num_hops, mode="all"))
    nodes = list(nb_u | nb_v)
    if len(nodes) > max_nodes:
        rng = rng or random
        core = {u, v}
        # Prefer keeping nodes shared between u's and v's neighborhoods
        shared_set = (nb_u & nb_v) - core
        shared = list(shared_set)
        rng.shuffle(shared)
        budget = max_nodes - len(core)
        shared_kept = shared[:budget]
        budget -= len(shared_kept)

        other = [n for n in nodes if n not in core and n not in shared_set]
        rng.shuffle(other)
        nodes = list(core) + shared_kept + other[:budget]

    sub = g.induced_subgraph(nodes)
    local = {orig: i for i, orig in enumerate(nodes)}
    u_local, v_local = local[u], local[v]

    # Remove only the target edge u -> v; a reverse edge v -> u stays
    to_delete = [e.index for e in sub.es if e.source == u_local and e.target == v_local]
    if to_delete:
        sub.delete_edges(to_delete)

    dist_u = sub.distances(source=[u_local], mode="out")[0]
    dist_v = sub.distances(source=[v_local], mode="in")[0]
    labels = [drnl_label(du, dv) for du, dv in zip(dist_u, dist_v)]

    if sub.ecount() == 0:
        edge_index = np.zeros((2, 0), dtype=np.int64)
    else:
        edge_index = np.array([[e.source, e.target] for e in sub.es], dtype=np.int64).T

    sub_x = node_features[nodes]
    return edge_index, np.array(labels, dtype=np.int64), sub_x


class SEALPairDataset(TorchDataset):
    """Lazily extracts each subgraph in __getitem__. Requires num_workers=0
    (self.rng is shared, not safe to pickle per worker)."""

    def __init__(self, g, node_features, num_hops, max_nodes, max_pairs, seed):
        super().__init__()
        self.g = g
        self.node_features = node_features
        self.num_hops = num_hops
        self.max_nodes = max_nodes
        self.rng = random.Random(seed)
        self.pairs, self.labels = self._build_pair_list(g, max_pairs, seed)

    @staticmethod
    def _build_pair_list(g, max_pairs, seed):
        """Samples positive pairs from real edges and negatives via negative_sampling()."""
        sources = [e.source for e in g.es]
        targets = [e.target for e in g.es]
        edge_index = torch.tensor([sources, targets], dtype=torch.long)
        edge_index = torch.unique(edge_index, dim=1)

        num_pos = min(max_pairs // 2, edge_index.size(1))
        perm = torch.randperm(edge_index.size(1))[:num_pos]
        pos_pairs = edge_index[:, perm].t().tolist()

        neg_index = negative_sampling(
            edge_index, num_nodes=g.vcount(), num_neg_samples=num_pos, method="sparse"
        )
        neg_pairs = neg_index.t().tolist()

        all_pairs = [(p, 1.0) for p in pos_pairs] + [(p, 0.0) for p in neg_pairs]
        pairs = [p for p, _ in all_pairs]
        labels = [y for _, y in all_pairs]
        return pairs, labels

    def len(self):
        return len(self.pairs)

    def get(self, idx):
        u, v = self.pairs[idx]
        y = self.labels[idx]
        sub_edge_index, node_labels, sub_x = extract_enclosing_subgraph(
            self.g, u, v, self.node_features, self.num_hops, self.max_nodes, self.rng
        )
        return Data(
            edge_index=torch.tensor(sub_edge_index, dtype=torch.long),
            node_label=torch.tensor(node_labels, dtype=torch.long),
            x=torch.tensor(sub_x, dtype=torch.float),
            num_nodes=len(node_labels),
            y=torch.tensor([y], dtype=torch.float),
        )


class SEALFixedPairDataset(TorchDataset):
    """Like SEALPairDataset, but for callers with their own fixed (u, v)
    pairs + labels instead of internally sampled pos/neg."""

    def __init__(self, g, node_features, num_hops, max_nodes, pairs, labels, seed, rng=None):
        super().__init__()
        self.g = g
        self.node_features = node_features
        self.num_hops = num_hops
        self.max_nodes = max_nodes
        self.rng = rng if rng is not None else random.Random(seed)
        self.pairs = pairs
        self.labels = labels

    def len(self):
        return len(self.pairs)

    def get(self, idx):
        u, v = self.pairs[idx]
        y = self.labels[idx]
        sub_edge_index, node_labels, sub_x = extract_enclosing_subgraph(
            self.g, u, v, self.node_features, self.num_hops, self.max_nodes, self.rng
        )
        return Data(
            edge_index=torch.tensor(sub_edge_index, dtype=torch.long),
            node_label=torch.tensor(node_labels, dtype=torch.long),
            x=torch.tensor(sub_x, dtype=torch.float),
            num_nodes=len(node_labels),
            y=torch.tensor([y], dtype=torch.float),
        )


class DGCNNPool(nn.Module):
    """Sort-pooling + 1-D CNN readout (DGCNN): turns a variable-size labeled
    subgraph into a fixed-size vector."""

    def __init__(self, in_dim, hidden_dim=32, num_gcn_layers=3, k=30):
        super().__init__()
        self.k = k
        dims = [in_dim] + [hidden_dim] * (num_gcn_layers - 1) + [1]
        self.convs = nn.ModuleList(
            GCNConv(dims[i], dims[i + 1]) for i in range(num_gcn_layers)
        )
        concat_dim = hidden_dim * (num_gcn_layers - 1) + 1

        self.conv1d_1 = nn.Conv1d(1, 16, concat_dim, concat_dim)
        self.pool1d = nn.MaxPool1d(2, 2)
        self.conv1d_2 = nn.Conv1d(16, 32, 5, 1)
        self.out_dim = ((k - 2) // 2 + 1 - 5 + 1) * 32

    def forward(self, x, edge_index, batch):
        xs = []
        h = x
        for conv in self.convs:
            h = torch.tanh(conv(h, edge_index))
            xs.append(h)
        h = torch.cat(xs, dim=-1)
        h = global_sort_pool(h, batch, self.k)
        h = h.view(h.size(0), 1, -1)
        h = F.relu(self.conv1d_1(h))
        h = self.pool1d(h)
        h = F.relu(self.conv1d_2(h))
        return h.view(h.size(0), -1)


class SEALLinkPredictor(nn.Module):
    """DGCNN input per node is [DRNL-label embedding || content features]."""

    def __init__(self, num_drnl_labels, label_dim, content_dim, hidden_dim=32, k=30):
        super().__init__()
        self.label_embed = nn.Embedding(num_drnl_labels + 1, label_dim)
        self.pool = DGCNNPool(label_dim + content_dim, hidden_dim=hidden_dim, k=k)
        self.classifier = nn.Sequential(
            nn.Linear(self.pool.out_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, batch):
        x = torch.cat([self.label_embed(batch.node_label), batch.x], dim=-1)
        subgraph_vec = self.pool(x, batch.edge_index, batch.batch)
        return self.classifier(subgraph_vec).squeeze(-1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_probs, all_y = [], []
    total_loss, n = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)
        total_loss += F.binary_cross_entropy_with_logits(logits, batch.y).item() * batch.num_graphs
        n += batch.num_graphs
        all_probs.append(torch.sigmoid(logits).cpu().numpy())
        all_y.append(batch.y.cpu().numpy())
    probs = np.concatenate(all_probs)
    y = np.concatenate(all_y)
    auc = roc_auc_score(y, probs)
    ap = average_precision_score(y, probs)
    avg_loss = total_loss / max(n, 1)
    return auc, ap, avg_loss


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

    node_features = {name: build_node_features(g) for name, g in graphs.items()}
    content_dim = next(iter(node_features.values())).shape[1]

    print(f"[phase] preparing candidate pairs (num_hops={args.num_hops}, "
          f"subgraphs extracted lazily per-batch)...")
    t0 = time.perf_counter()
    train_set = SEALPairDataset(graphs["train"], node_features["train"], args.num_hops,
                                 args.max_nodes, args.max_train_pairs, args.seed)
    dev_set = SEALPairDataset(graphs["dev"], node_features["dev"], args.num_hops,
                               args.max_nodes, args.max_eval_pairs, args.seed)
    test_set = SEALPairDataset(graphs["test"], node_features["test"], args.num_hops,
                                args.max_nodes, args.max_eval_pairs, args.seed)
    _phase("preparing candidate pairs", t0)
    print(f"train/dev/test pairs: {len(train_set)}/{len(dev_set)}/{len(test_set)}")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    dev_loader = DataLoader(dev_set, batch_size=args.batch_size)
    test_loader = DataLoader(test_set, batch_size=args.batch_size)

    model = SEALLinkPredictor(MAX_DRNL_LABEL, args.label_dim, content_dim,
                               hidden_dim=args.hidden_dim, k=args.sort_k).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    epoch_rows = []  # (epoch, train_loss, dev_loss, dev_auc, dev_ap, elapsed_s)
    ckpt_exists = os.path.exists(out_path)
    val_metric_name = args.val_metric

    if ckpt_exists:
        print(f"[phase] {out_path} already exists -- skipping training, "
              f"loading checkpoint for validation/test only")
        model.load_state_dict(torch.load(out_path, map_location=device))
        dev_auc, dev_ap, _ = evaluate(model, dev_loader, device)
        best_dev_metric = dev_auc if val_metric_name == "auc" else dev_ap
    else:
        print(f"[phase] training for up to {args.epochs} epochs "
              f"(model selection + early stopping on dev_{val_metric_name}, patience={args.patience})...")
        t0 = time.perf_counter()
        stopper = EarlyStopper(args.patience, args.min_delta)
        for epoch in range(1, args.epochs + 1):
            model.train()
            total_loss, n = 0.0, 0
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                logits = model(batch)
                loss = F.binary_cross_entropy_with_logits(logits, batch.y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * batch.num_graphs
                n += batch.num_graphs

            dev_auc, dev_ap, dev_loss = evaluate(model, dev_loader, device)
            dev_metric = dev_auc if val_metric_name == "auc" else dev_ap
            elapsed = time.perf_counter() - t0
            avg_loss = total_loss / max(n, 1)
            print(f"epoch {epoch:03d}  train_loss={avg_loss:.4f}  dev_loss={dev_loss:.4f}  "
                  f"dev_auc={dev_auc:.4f}  dev_ap={dev_ap:.4f}  ({elapsed:.1f}s elapsed)")
            epoch_rows.append((epoch, avg_loss, dev_loss, dev_auc, dev_ap, elapsed))

            if stopper.update(dev_metric):
                torch.save(model.state_dict(), out_path)
            elif stopper.should_stop():
                print(f"[phase] early stopping: dev_{val_metric_name} hasn't improved by >= "
                      f"{args.min_delta} for {args.patience} epochs (best={stopper.best_metric:.4f})")
                break
        best_dev_metric = stopper.best_metric
        _phase(f"training ({len(epoch_rows)} epochs)", t0)

        epoch_csv = os.path.join(out_dir or RESULTS_DIR, f"seal_epoch_metrics{tag}.csv")
        write_epoch_csv(epoch_csv, epoch_rows)
        print(f"[phase] per-epoch metrics written to {epoch_csv}")

    print(f"\nBest dev {val_metric_name}: {best_dev_metric:.4f} (checkpoint: {out_path})")
    model.load_state_dict(torch.load(out_path, map_location=device))

    print("[phase] evaluating held-out dev/test...")
    t0 = time.perf_counter()
    dev_auc, dev_ap, _ = evaluate(model, dev_loader, device)
    test_auc, test_ap, _ = evaluate(model, test_loader, device)
    _phase("evaluating held-out dev/test", t0)
    print(f"\nDev AUC: {dev_auc:.4f}  Dev AP: {dev_ap:.4f}")
    print(f"Test AUC: {test_auc:.4f}  Test AP: {test_ap:.4f}")

    eval_csv = os.path.join(out_dir or RESULTS_DIR, f"seal_eval_metrics{tag}.csv")
    write_eval_csv(eval_csv, [
        ("dev", "inductive (model selection)", dev_auc, dev_ap),
        ("test", "inductive", test_auc, test_ap),
    ])
    print(f"[phase] held-out eval metrics written to {eval_csv}")
    _phase("total run time", run_start)

    return {"seed": args.seed, "dev_auc": dev_auc, "dev_ap": dev_ap, "test_auc": test_auc, "test_ap": test_ap}


def main():
    parser = argparse.ArgumentParser(description="Train a SEAL link-existence predictor on the SciCite citation graph")
    parser.add_argument("--num-hops", type=int, default=1, help="BFS radius around each candidate pair's endpoints")
    parser.add_argument("--max-nodes", type=int, default=100, help="cap on enclosing subgraph size (hub-node safeguard)")
    parser.add_argument("--sort-k", type=int, default=30, help="SortPooling: number of nodes kept per subgraph")
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--label-dim", type=int, default=16, help="DRNL label embedding size")
    parser.add_argument("--max-train-pairs", type=int, default=20000, help="positive+negative pairs; halved for each class")
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
                              "are written to GNNs/results/seal_eval_seed.csv, one row per seed")
    parser.add_argument("--seed-dir", type=str, default=None,
                         help="with --seeds, write all per-seed checkpoints/epoch-metrics/eval-metrics "
                              "and the final seal_eval_seed.csv into this folder (created if needed) "
                              "instead of scattering them across the checkpoint's own directory and "
                              "GNNs/results")
    parser.add_argument("--out", type=str, default="seal_link_predictor.pt")
    args = parser.parse_args()

    if args.seeds:
        run_multi_seed(
            args, run, "seal_eval_seed.csv",
            ["seed", "dev_auc", "dev_ap", "test_auc", "test_ap"],
            RESULTS_DIR,
        )
    else:
        run(args)


if __name__ == "__main__":
    main()
