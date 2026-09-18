"""Metric definitions matching scripts/evaluate_verdict_baseline.py, plus the second ECE channel."""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np


def ece(confidence: Sequence[float], correct: Sequence[float], bins: int = 10) -> float:
    """Expected calibration error with equal-width bins, as the benchmark harness computes it."""
    conf = np.asarray(confidence, dtype=float)
    acc = np.asarray(correct, dtype=float)
    if conf.size == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        sel = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if sel.sum():
            total += sel.mean() * abs(acc[sel].mean() - conf[sel].mean())
    return float(total)


def summarize(records: List[Dict]) -> Dict[str, float]:
    """Aggregate per-decision records into the full reported metric set.

    Each record needs: probs, target, label_index, qtype, confidence, expected_level, gold_score.
    Brier covers choice and noul only, matching the benchmark harness.
    """
    acc, briers, soft, maes, within, dist_conf, head_conf = [], [], [], [], [], [], []

    for r in records:
        p = np.asarray(r["probs"], dtype=float)
        g = np.asarray(r["target"], dtype=float)
        hit = float(int(np.argmax(p)) == int(r["label_index"]))
        acc.append(hit)
        dist_conf.append(float(p.max()))
        head_conf.append(float(r["confidence"]))
        if r["qtype"] != 1:  # choice and noul
            briers.append(float(((p - g) ** 2).sum()))
            soft.append(float((p * g).sum()))
        else:
            err = abs(float(r["expected_level"]) - float(r["gold_score"]))
            maes.append(err)
            within.append(float(err <= 1.0))

    return {
        "n": len(records),
        "accuracy": float(np.mean(acc)) if acc else 0.0,
        "soft_accuracy": float(np.mean(soft)) if soft else 0.0,
        "brier": float(np.mean(briers)) if briers else 0.0,
        "ece_distribution": ece(dist_conf, acc),
        "ece_confidence": ece(head_conf, acc),
        "score_mae": float(np.mean(maes)) if maes else 0.0,
        "within_one_level": float(np.mean(within)) if within else 0.0,
        "mean_distribution_confidence": float(np.mean(dist_conf)) if dist_conf else 0.0,
        "mean_head_confidence": float(np.mean(head_conf)) if head_conf else 0.0,
    }
