"""Shared text-embedding helpers used by embed_scibert.py and
second_hand_citation.py."""
import torch


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean-pools a transformer's last hidden state over real (non-padding)
    tokens, per attention_mask. Mean pooling generalizes better than
    [CLS]-only for similarity-style downstream use, and is the common
    choice for SciBERT-based retrieval."""
    mask = attention_mask.unsqueeze(-1).float()
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts
