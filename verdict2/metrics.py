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


def auroc(scores: Sequence[float], correct: Sequence[float]) -> float:
    """Calculate Area Under the ROC Curve via vectorized Mann-Whitney U rank statistic."""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(correct, dtype=float)
    pos = s[y == 1]
    neg = s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # Exact rank statistic: P(score_pos > score_neg) + 0.5 * P(score_pos == score_neg)
    u = np.sum(pos[:, None] > neg[None, :]) + 0.5 * np.sum(pos[:, None] == neg[None, :])
    return float(u / (len(pos) * len(neg)))


def selective_risk_coverage(
    scores: Sequence[float],
    correct: Sequence[float],
    coverages: Sequence[float] = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3),
) -> List[Dict[str, float]]:
    """Calculate retained accuracy and selective risk across traffic coverage targets."""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(correct, dtype=float)
    n = len(s)
    if n == 0:
        return []
    order = np.argsort(-s)
    sorted_y = y[order]
    sorted_s = s[order]

    results = []
    for cov in coverages:
        k = max(1, int(round(cov * n)))
        actual_cov = float(k / n)
        retained_y = sorted_y[:k]
        retained_acc = float(np.mean(retained_y))
        risk = float(1.0 - retained_acc)
        min_threshold = float(sorted_s[k - 1])
        # Wilson 95% CI upper bound on selective risk
        z = 1.96
        denom = 1.0 + (z**2) / k
        centre = (risk + (z**2) / (2.0 * k)) / denom
        spread = (z * np.sqrt(max(0.0, risk * (1.0 - risk) + (z**2) / (4.0 * k)) / k)) / denom
        risk_ub_95 = float(min(1.0, centre + spread))

        results.append({
            "target_coverage": float(cov),
            "actual_coverage": round(actual_cov, 4),
            "retained_accuracy": round(retained_acc, 4),
            "selective_risk": round(risk, 4),
            "risk_upper_bound_95": round(risk_ub_95, 4),
            "threshold": round(min_threshold, 4),
            "accepted_count": k,
            "errors": int(np.sum(retained_y == 0)),
        })
    return results


def summarize(records: List[Dict]) -> Dict[str, Any]:
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

    curve = selective_risk_coverage(head_conf, acc)
    curve_by_cov = {f"acc_at_{int(round(c['target_coverage']*100))}_coverage": c["retained_accuracy"] for c in curve}

    return {
        "n": len(records),
        "accuracy": float(np.mean(acc)) if acc else 0.0,
        "soft_accuracy": float(np.mean(soft)) if soft else 0.0,
        "brier": float(np.mean(briers)) if briers else 0.0,
        "ece_distribution": ece(dist_conf, acc),
        "ece_confidence": ece(head_conf, acc),
        "auroc_confidence": auroc(head_conf, acc),
        "auroc_distribution": auroc(dist_conf, acc),
        "score_mae": float(np.mean(maes)) if maes else 0.0,
        "within_one_level": float(np.mean(within)) if within else 0.0,
        "mean_distribution_confidence": float(np.mean(dist_conf)) if dist_conf else 0.0,
        "mean_head_confidence": float(np.mean(head_conf)) if head_conf else 0.0,
        **curve_by_cov,
        "selective_classification": curve,
    }
