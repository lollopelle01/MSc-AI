"""Re-evaluates the trained GraphSAGE and SEAL link-existence predictors
against hard negatives (open 2-hop triads: u cites w, w cites v, u->v not
a real edge) instead of the uniformly-random ones used at training time.
Loads existing checkpoints; doesn't retrain anything."""
import argparse
import random

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import RandomLinkSplit

from data_utils import build_node_features, igraph_to_pyg_data, load_split_graphs
from hard_negatives import build_hard_negative_candidates
from train_graphsage import GraphSAGELinkPredictor
from train_seal import MAX_DRNL_LABEL, SEALLinkPredictor, SEALFixedPairDataset


def evaluate_graphsage(graphs, ckpt_path, splits, max_hard_negatives, max_per_hub,
                        hidden_dim, num_layers, val_frac, test_frac, seed, device):
    print("\n=== GraphSAGE: hard-negative evaluation ===")
    data = {name: igraph_to_pyg_data(g) for name, g in graphs.items()}
    in_dim = data[splits[0]].x.size(-1)
    model = GraphSAGELinkPredictor(in_dim, hidden_dim, num_layers).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    # neg_sampling_ratio=0: edge_label_index holds only held-out positives.
    splitter = RandomLinkSplit(num_val=val_frac, num_test=test_frac, is_undirected=False,
                                add_negative_train_samples=False, neg_sampling_ratio=0.0)

    for name in splits:
        g = graphs[name]
        torch.manual_seed(seed)
        _, _, test_split = splitter(data[name])
        pos_pairs = test_split.edge_label_index

        hard_neg_pairs = build_hard_negative_candidates(g, max_hard_negatives, max_per_hub, seed)
        if not hard_neg_pairs:
            print(f"{name}: no hard-negative candidates found, skipping")
            continue

        # Balance 1:1 so AP is comparable across splits/models.
        n = min(pos_pairs.size(1), len(hard_neg_pairs))
        rng = random.Random(seed)
        pos_pairs = pos_pairs[:, rng.sample(range(pos_pairs.size(1)), n)]
        hard_neg_pairs = rng.sample(hard_neg_pairs, n)
        neg_tensor = torch.tensor(hard_neg_pairs, dtype=torch.long).t()

        edge_label_index = torch.cat([pos_pairs, neg_tensor], dim=1).to(device)
        edge_label = torch.cat([
            torch.ones(pos_pairs.size(1)), torch.zeros(neg_tensor.size(1)),
        ]).to(device)
        x = test_split.x.to(device)
        message_edge_index = test_split.edge_index.to(device)

        with torch.no_grad():
            logits = model(x, message_edge_index, edge_label_index)
        probs = torch.sigmoid(logits).cpu().numpy()
        y = edge_label.cpu().numpy()
        auc = roc_auc_score(y, probs)
        ap = average_precision_score(y, probs)
        print(f"{name}: hard-negative AUC={auc:.4f}  AP={ap:.4f}  "
              f"(n_pos={pos_pairs.size(1)}, n_hard_neg={neg_tensor.size(1)})")


def evaluate_seal(graphs, ckpt_path, splits, max_pairs, num_hops, max_nodes,
                   sort_k, hidden_dim, label_dim, seed, device):
    print("\n=== SEAL: hard-negative evaluation ===")
    node_features = {name: build_node_features(g) for name, g in graphs.items()}
    content_dim = next(iter(node_features.values())).shape[1]
    model = SEALLinkPredictor(MAX_DRNL_LABEL, label_dim, content_dim,
                               hidden_dim=hidden_dim, k=sort_k).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    rng = random.Random(seed)
    for name in splits:
        g = graphs[name]

        hard_neg_pool = build_hard_negative_candidates(g, max_total=max(max_pairs, 2000),
                                                        max_per_hub=50, seed=seed)
        rng.shuffle(hard_neg_pool)
        neg_pairs = hard_neg_pool[:max_pairs]

        edge_pairs = list(set(g.get_edgelist()))
        rng.shuffle(edge_pairs)
        pos_pairs = edge_pairs[:len(neg_pairs)]
        if len(neg_pairs) < max_pairs:
            print(f"{name}: only found {len(neg_pairs)} hard negatives "
                  f"(wanted {max_pairs}); matching positives to that count")

        # Lazy extraction (SEALFixedPairDataset) keeps subgraph building out
        # of RAM until scoring time. rng continues the same state used by
        # the shuffles above rather than a fresh one from `seed`.
        pairs = pos_pairs + neg_pairs
        labels = [1.0] * len(pos_pairs) + [0.0] * len(neg_pairs)
        dataset = SEALFixedPairDataset(g, node_features[name], num_hops, max_nodes,
                                        pairs, labels, seed, rng=rng)

        loader = DataLoader(dataset, batch_size=64)
        all_probs, all_y = [], []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                logits = model(batch)
                all_probs.append(torch.sigmoid(logits).cpu().numpy())
                all_y.append(batch.y.cpu().numpy())
        probs = np.concatenate(all_probs)
        y = np.concatenate(all_y)
        auc = roc_auc_score(y, probs)
        ap = average_precision_score(y, probs)
        print(f"{name}: hard-negative AUC={auc:.4f}  AP={ap:.4f}  "
              f"(n_pos={len(pos_pairs)}, n_hard_neg={len(neg_pairs)})")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained GraphSAGE/SEAL checkpoints against hard "
                    "(2-hop, structurally-plausible) negatives instead of random ones."
    )
    parser.add_argument("--splits", nargs="+", default=["dev", "test"])
    # Defaults (checkpoint path + architecture dims) must match the
    # notebook's final training run, not train_graphsage.py's/train_seal.py's
    # own defaults, or load_state_dict below fails on a shape mismatch.
    parser.add_argument("--graphsage-ckpt", type=str,
                         default="../GNNs/weights/graphsage_link_predictor_es_ap_final.pt")
    parser.add_argument("--seal-ckpt", type=str,
                         default="../GNNs/weights/seal_link_predictor_es_ap_final.pt")
    parser.add_argument("--skip-graphsage", action="store_true")
    parser.add_argument("--skip-seal", action="store_true")
    parser.add_argument("--max-hard-negatives", type=int, default=20000,
                         help="GraphSAGE: hard-negative pool size per split")
    parser.add_argument("--max-per-hub", type=int, default=50,
                         help="cap on candidates contributed by a single intermediate paper")
    parser.add_argument("--hidden-dim", type=int, default=256,
                         help="GraphSAGE hidden dim -- must match the trained checkpoint "
                              "(notebook section 2 final run uses 256, not train_graphsage.py's own default of 128)")
    parser.add_argument("--num-layers", type=int, default=2,
                         help="GraphSAGE layers -- must match the trained checkpoint")
    parser.add_argument("--val-frac", type=float, default=0.05)
    parser.add_argument("--test-frac", type=float, default=0.10)
    parser.add_argument("--seal-max-pairs", type=int, default=1000,
                         help="SEAL: positives+negatives per split (subgraph extraction is expensive)")
    parser.add_argument("--seal-num-hops", type=int, default=1)
    parser.add_argument("--seal-max-nodes", type=int, default=100)
    parser.add_argument("--seal-sort-k", type=int, default=30)
    parser.add_argument("--seal-hidden-dim", type=int, default=64,
                         help="must match the trained checkpoint (notebook section 3 final run uses 64, "
                              "not train_seal.py's own default of 32)")
    parser.add_argument("--seal-label-dim", type=int, default=32,
                         help="must match the trained checkpoint (notebook section 3 final run uses 32, "
                              "not train_seal.py's own default of 16)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)

    print("Loading split graphs (train/dev/test)...")
    graphs = load_split_graphs()

    if not args.skip_graphsage:
        evaluate_graphsage(graphs, args.graphsage_ckpt, args.splits, args.max_hard_negatives,
                            args.max_per_hub, args.hidden_dim, args.num_layers,
                            args.val_frac, args.test_frac, args.seed, device)

    if not args.skip_seal:
        evaluate_seal(graphs, args.seal_ckpt, args.splits, args.seal_max_pairs,
                       args.seal_num_hops, args.seal_max_nodes, args.seal_sort_k,
                       args.seal_hidden_dim, args.seal_label_dim, args.seed, device)


if __name__ == "__main__":
    main()
