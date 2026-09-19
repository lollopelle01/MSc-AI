"""Shared hard-negative evaluation helpers used by train_graphsage_hard.py
and mlp_baseline_hard_common.py."""
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

from hard_negatives import sample_matched_hard_negatives


def cap_pairs(edge_label_index, max_pairs, rng):
    """Subsamples edge_label_index (shape [2, N]) down to max_pairs columns,
    for budget-matched comparisons against train_seal_hard.py."""
    n = edge_label_index.size(1)
    if max_pairs is None or max_pairs >= n:
        return edge_label_index
    idx = rng.sample(range(n), max_pairs)
    return edge_label_index[:, idx]


def _batched_score(model, h, edge_label_index, batch_size):
    """Scores edge_label_index through model's decoder attribute
    (model.predictor for GraphSAGELinkPredictor, model.decoder for
    MLPBaselineLinkPredictor) in chunks of `batch_size` pairs, concatenating
    results back in order. No-op wrapper when batch_size is None or >= n."""
    decoder = model.predictor if hasattr(model, "predictor") else model.decoder
    n = edge_label_index.size(1)
    if batch_size is None or batch_size >= n:
        return decoder(h, edge_label_index)
    return torch.cat([
        decoder(h, edge_label_index[:, start:start + batch_size])
        for start in range(0, n, batch_size)
    ], dim=0)


def _encode(model, split_data):
    """Runs model's encoder pass once. SAGEEncoder.forward(x, edge_index)
    does message passing; NodeMLPEncoder.forward(x) is purely per-node;
    model.encoder is None for the "raw" MLP baseline variant (h = x)."""
    if model.encoder is None:
        return split_data.x
    if hasattr(model, "predictor"):  # GraphSAGELinkPredictor
        return model.encoder(split_data.x, split_data.edge_index)
    return model.encoder(split_data.x)  # MLPBaselineLinkPredictor, variant="encoded"


@torch.no_grad()
def evaluate_hard(model, split_data, two_hop_targets, num_nodes, rng, device, decoder_batch_size=None):
    """AUC/AP on split_data's held-out positives vs. one matched hard
    negative per positive's anchor (falls back to random when an anchor
    has no 2-hop candidate). Works for any model whose forward signature is
    model(x, edge_index, edge_label_index) -> logits."""
    model.eval()
    split_data = split_data.to(device)
    pos_pairs = split_data.edge_label_index
    sources = pos_pairs[0].tolist()
    neg_targets, n_hard, n_fallback = sample_matched_hard_negatives(sources, two_hop_targets, num_nodes, rng)
    neg_pairs = torch.tensor([sources, neg_targets], dtype=torch.long, device=device)

    edge_label_index = torch.cat([pos_pairs, neg_pairs], dim=1)
    edge_label = torch.cat([
        torch.ones(pos_pairs.size(1), device=device), torch.zeros(neg_pairs.size(1), device=device),
    ])

    h = _encode(model, split_data)
    logits = _batched_score(model, h, edge_label_index, decoder_batch_size)
    n_pos = pos_pairs.size(1)
    loss = F.softplus(logits[n_pos:] - logits[:n_pos]).mean().item()
    probs = torch.sigmoid(logits).cpu().numpy()
    y = edge_label.cpu().numpy()
    auc = roc_auc_score(y, probs)
    ap = average_precision_score(y, probs)
    return auc, ap, loss, n_hard, n_fallback
