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


def permutation_kl(
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
    orders: list[list[int]],
) -> torch.Tensor:
    """Symmetric KL between the same questions scored under two option orderings.

    Borrowed from Kev, which trains this term and still measures a 7.41% argmax flip rate under
    option reordering. `orders[i][j]` is the original option index now at position j in the twin,
    so the twin's distribution is scattered back onto the original ordering before comparison.
    """
    log_a = torch.log_softmax(logits_a, dim=-1)
    log_b_raw = torch.log_softmax(logits_b, dim=-1)

    log_b = torch.full_like(log_a, -1e4)
    for i, order in enumerate(orders):
        index = torch.tensor(order, device=log_a.device, dtype=torch.long)
        log_b[i, index] = log_b_raw[i, : len(order)]

    kl_ab = F.kl_div(log_b, log_a, log_target=True, reduction="none").sum(-1)
    kl_ba = F.kl_div(log_a, log_b, log_target=True, reduction="none").sum(-1)
    return (0.5 * (kl_ab + kl_ba)).mean()


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
