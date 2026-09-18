"""Strictly proper scoring loss functions, temperature scaling, and calibration metrics.

Implements composite Cross-Entropy + Brier score loss, positive scalar
temperature scaling fitted via L-BFGS, and 10-bin equal-width/equal-mass ECE.
Never clamps probabilities or invents heuristic confidence fallbacks.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def cross_entropy_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    reduction: Literal["mean", "none"] = "mean",
) -> torch.Tensor:
    """Compute standard categorical negative log-likelihood (NLL) loss.

    Args:
        logits: Unnormalized predictions of shape (batch_size, num_classes).
        targets: Target class indices of shape (batch_size,).
        valid_mask: Optional boolean mask of shape (batch_size, num_classes)
            indicating valid candidate options. Invalid options are masked with -inf.
        reduction: 'mean' across batch or 'none'.
    """
    if not torch.all(torch.isfinite(logits)):
        raise ValueError("Logits contain non-finite values (NaN or Inf).")

    if valid_mask is not None:
        masked_logits = logits.masked_fill(~valid_mask, float("-inf"))
    else:
        masked_logits = logits

    log_probs = F.log_softmax(masked_logits, dim=-1)
    loss = F.nll_loss(log_probs, targets, reduction=reduction)
    return loss


def brier_score_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    reduction: Literal["mean", "none"] = "mean",
) -> torch.Tensor:
    """Compute summed multiclass Brier score loss: \\sum_{k=1}^K (p_k - y_k)^2.

    The multiclass Brier score is a strictly proper scoring rule with range [0, 2].
    """
    if not torch.all(torch.isfinite(logits)):
        raise ValueError("Logits contain non-finite values (NaN or Inf).")

    if valid_mask is not None:
        masked_logits = logits.masked_fill(~valid_mask, float("-inf"))
    else:
        masked_logits = logits

    probs = F.softmax(masked_logits, dim=-1)
    # Zero out probabilities on masked candidates to prevent nan in gradients
    if valid_mask is not None:
        probs = probs.masked_fill(~valid_mask, 0.0)

    num_classes = logits.shape[-1]
    one_hot = F.one_hot(targets, num_classes=num_classes).to(probs.dtype)

    # Sum squared differences across classes
    squared_diff = torch.sum((probs - one_hot) ** 2, dim=-1)

    if reduction == "mean":
        return torch.mean(squared_diff)
    return squared_diff


def composite_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    lambda_brier: float = 1.0,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute composite loss: L = L_CE + lambda * L_Brier."""
    ce = cross_entropy_loss(logits, targets, valid_mask=valid_mask, reduction="mean")
    brier = brier_score_loss(logits, targets, valid_mask=valid_mask, reduction="mean")
    return ce + lambda_brier * brier


@dataclass(frozen=True)
class ECEResult:
    """Calibration error statistics and bin details."""

    ece: float
    mce: float
    bin_accuracies: tuple[float, ...]
    bin_confidences: tuple[float, ...]
    bin_counts: tuple[int, ...]
    bin_edges: tuple[float, ...]
    strategy: str
    min_bin_count: int = 10


def compute_ece(
    confidences: np.ndarray,
    accuracies: np.ndarray,
    n_bins: int = 10,
    strategy: Literal["equal_width", "equal_mass", "adaptive"] = "equal_width",
    min_bin_count: int = 10,
) -> ECEResult:
    """Compute Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).

    Args:
        confidences: 1D array of predicted confidence values in [0, 1].
        accuracies: 1D binary array of correctness (1 for correct, 0 for incorrect).
        n_bins: Number of bins (standard is 10).
        strategy: 'equal_width', 'equal_mass', or 'adaptive'.
        min_bin_count: Minimum samples required in a bin to include it in MCE calculation.
    """
    n_samples = len(confidences)
    if n_samples == 0:
        return ECEResult(
            ece=0.0,
            mce=0.0,
            bin_accuracies=(),
            bin_confidences=(),
            bin_counts=(),
            bin_edges=(),
            strategy=strategy,
            min_bin_count=min_bin_count,
        )

    if strategy == "equal_width":
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        bin_assignments = np.digitize(confidences, bin_edges[1:-1])  # 0 to n_bins - 1
    elif strategy == "equal_mass":
        quantiles = np.linspace(0.0, 1.0, n_bins + 1)
        bin_edges = np.percentile(confidences, quantiles * 100)
        bin_edges[0] = 0.0
        bin_edges[-1] = 1.0
        bin_assignments = np.digitize(confidences, bin_edges[1:-1])
    elif strategy == "adaptive":
        sorted_indices = np.argsort(confidences)
        bin_assignments = np.zeros(n_samples, dtype=int)
        bin_edges_list = [0.0]
        cur_bin = 0
        cur_count = 0
        target_bin_size = max(min_bin_count, n_samples // n_bins)

        for i, idx in enumerate(sorted_indices):
            cur_count += 1
            bin_assignments[idx] = cur_bin
            if cur_count >= target_bin_size and (n_samples - i - 1) >= min_bin_count:
                bin_edges_list.append(float(confidences[idx]))
                cur_bin += 1
                cur_count = 0
        bin_edges_list.append(1.0)
        bin_edges = np.array(bin_edges_list)
        n_bins = cur_bin + 1
    else:
        raise ValueError(f"Unknown ECE strategy: {strategy!r}")

    bin_accs: list[float] = []
    bin_confs: list[float] = []
    bin_counts: list[int] = []
    ece = 0.0
    mce = 0.0

    for b in range(n_bins):
        mask = bin_assignments == b
        count = int(np.sum(mask))
        bin_counts.append(count)

        if count > 0:
            bin_acc = float(np.mean(accuracies[mask]))
            bin_conf = float(np.mean(confidences[mask]))
            bin_accs.append(bin_acc)
            bin_confs.append(bin_conf)

            diff = abs(bin_acc - bin_conf)
            ece += (count / n_samples) * diff
            if count >= min_bin_count and diff > mce:
                mce = diff
        else:
            bin_accs.append(0.0)
            bin_confs.append(0.0)

    return ECEResult(
        ece=float(ece),
        mce=float(mce),
        bin_accuracies=tuple(bin_accs),
        bin_confidences=tuple(bin_confs),
        bin_counts=tuple(bin_counts),
        bin_edges=tuple(float(e) for e in bin_edges),
        strategy=strategy,
        min_bin_count=min_bin_count,
    )


def compute_bootstrap_ci(
    values: np.ndarray,
    metric_fn: Any,
    n_resamples: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Compute empirical bootstrap confidence interval for any metric function."""
    rng = np.random.default_rng(seed)
    n = len(values)
    if n == 0:
        return (0.0, 0.0)

    boot_metrics = []
    for _ in range(n_resamples):
        indices = rng.integers(0, n, size=n)
        sample = values[indices]
        boot_metrics.append(metric_fn(sample))

    alpha = (1.0 - ci) / 2.0
    low = float(np.percentile(boot_metrics, alpha * 100))
    high = float(np.percentile(boot_metrics, (1.0 - alpha) * 100))
    return (low, high)


class TemperatureCalibrator(nn.Module):
    """Post-hoc probability calibration using positive scalar temperature T = exp(theta).

    Optimized via L-BFGS on held-out calibration logits and targets.
    Preserves exact argmax rankings while minimizing negative log-likelihood.
    Binds explicitly to evaluated model identity, artifact hash, and schema scope.
    """

    def __init__(
        self,
        model_id: str = "default",
        initial_temperature: float = 1.0,
        scope: str = "restricted_5_candidate_selection",
        artifact_hash: str = "",
    ):
        super().__init__()
        self.model_id = model_id
        self.scope = scope
        self.artifact_hash = artifact_hash
        if initial_temperature <= 0.0:
            raise ValueError("Initial temperature must be strictly positive.")
        # Parameterized as log_temperature: T = exp(log_temperature) > 0 strictly
        self.log_temperature = nn.Parameter(
            torch.tensor([np.log(initial_temperature)], dtype=torch.float32)
        )

    @property
    def temperature(self) -> float:
        return float(torch.exp(self.log_temperature).item())

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply scalar temperature scaling to raw logits."""
        if not torch.all(torch.isfinite(logits)):
            raise ValueError("Logits contain non-finite values (NaN or Inf).")
        temp = torch.exp(self.log_temperature).to(logits.device)
        return logits / temp

    def fit(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        max_iter: int = 50,
        lr: float = 0.05,
    ) -> TemperatureCalibrator:
        """Fit optimal positive temperature on calibration logits via L-BFGS."""
        if not torch.all(torch.isfinite(logits)):
            raise ValueError("Calibration logits contain non-finite values.")

        logits_frozen = logits.detach().float()
        targets_frozen = targets.detach()
        mask_frozen = valid_mask.detach() if valid_mask is not None else None

        optimizer = torch.optim.LBFGS(
            [self.log_temperature],
            lr=lr,
            max_iter=max_iter,
            line_search_fn="strong_wolfe",
        )

        def closure():
            optimizer.zero_grad()
            scaled_logits = self.forward(logits_frozen)
            loss = cross_entropy_loss(
                scaled_logits, targets_frozen, valid_mask=mask_frozen
            )
            loss.backward()
            return loss

        optimizer.step(closure)
        return self

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "temperature": self.temperature,
            "log_temperature": float(self.log_temperature.item()),
            "scope": self.scope,
            "artifact_hash": self.artifact_hash,
            "format_version": "rlcd-calibrator-v1",
        }

    @classmethod
    def from_dict(cls, data: dict) -> TemperatureCalibrator:
        calibrator = cls(
            model_id=data.get("model_id", "default"),
            initial_temperature=float(data["temperature"]),
            scope=data.get("scope", "restricted_5_candidate_selection"),
            artifact_hash=data.get("artifact_hash", ""),
        )
        return calibrator

    def save(self, file_path: str | Path) -> None:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, file_path: str | Path) -> TemperatureCalibrator:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
