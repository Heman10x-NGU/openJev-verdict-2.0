"""Typed loss routing against soft teacher targets.

The benchmark's Brier metric compares the predicted distribution to the teacher panel's
distribution, so the training objective targets that distribution directly. A small hard-label
term recovers the 1.55% of items whose `label` differs from `argmax(probabilities)`.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

SCORE_QTYPE = 1


def soft_cross_entropy(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    log_p = torch.log_softmax(logits, dim=-1)
    return -(target * log_p * mask).sum(-1)


def brier(probs: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (((probs - target) ** 2) * mask).sum(-1)


def ranked_probability_score(probs: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Squared distance between predicted and teacher CDFs, for ordered levels."""
    k = mask.sum(-1).clamp(min=2).float()
    diff = (torch.cumsum(probs * mask, -1) - torch.cumsum(target * mask, -1)) ** 2
    return (diff * mask).sum(-1) / (k - 1)


def decision_loss(
    logits: torch.Tensor,
    batch: dict,
    lambda_brier: float = 0.5,
    lambda_hard: float = 0.25,
    lambda_rps: float = 1.0,
) -> torch.Tensor:
    """Composite strictly-proper objective, routed by question type."""
    mask = batch["marker_mask"].float()
    target = batch["target"]
    probs = torch.softmax(logits, dim=-1)

    loss = soft_cross_entropy(logits, target, mask)
    if lambda_brier > 0:
        loss = loss + lambda_brier * brier(probs, target, mask)
    if lambda_hard > 0:
        loss = loss + lambda_hard * F.cross_entropy(logits, batch["label"], reduction="none")
    if lambda_rps > 0:
        is_score = (batch["qtype"] == SCORE_QTYPE).float()
        loss = loss + lambda_rps * is_score * ranked_probability_score(probs, target, mask)
    return loss.mean()
