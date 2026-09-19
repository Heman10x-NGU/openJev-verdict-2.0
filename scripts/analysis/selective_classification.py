#!/usr/bin/env python3
"""Produce the receipt behind the AUROC and selective-classification claims.

The README and the dashboard quote AUROC 0.7861 and accuracy at 80% and 60% coverage, but no JSON in
this repository computes them. This script does, so the numbers stop being prose.

    python scripts/analysis/selective_classification.py \
        --checkpoint artifacts/verdict2-base/model.pt \
        --out reports/verdict2_selective.json

Quote whatever this prints. If it disagrees with the numbers currently on the page, the page is wrong.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from verdict2.data import load_items
from verdict2.evaluate import TOKENIZER
from verdict2.model import VerdictModel
from verdict2.train import infer, pick_device


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based AUROC. Equivalent to the Mann-Whitney U statistic, ties averaged."""
    positives, negatives = labels.sum(), (1 - labels).sum()
    if positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks within tied score groups
    unique, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    for i, count in enumerate(counts):
        if count > 1:
            mask = inverse == i
            ranks[mask] = ranks[mask].mean()
    return float((ranks[labels == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def selective_curve(confidence: np.ndarray, correct: np.ndarray, coverages) -> list:
    """Accuracy retained when you keep only the most confident fraction of decisions."""
    order = np.argsort(-confidence)
    rows = []
    for coverage in coverages:
        keep = max(1, int(round(coverage * len(confidence))))
        selected = order[:keep]
        rows.append(
            {
                "coverage": round(keep / len(confidence), 4),
                "retained_accuracy": round(float(correct[selected].mean()), 4),
                "selective_risk": round(float(1 - correct[selected].mean()), 4),
                "confidence_threshold": round(float(confidence[selected][-1]), 4),
                "accepted": int(keep),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="AUROC and selective classification for Verdict 2.0.")
    parser.add_argument("--checkpoint", default="artifacts/verdict2-base/model.pt")
    parser.add_argument("--out", default="reports/verdict2_selective.json")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = pick_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    backbone = checkpoint["backbone"]

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(backbone)
    import verdict2.evaluate as ev

    ev.TOKENIZER = tokenizer
    pad_id = tokenizer.pad_token_id or 0

    model = VerdictModel(backbone).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    items = load_items("test", tokenizer)
    records = infer(model, items, pad_id, device, args.batch_size, use_temperature=True)

    correct = np.array([float(int(np.argmax(r["probs"])) == int(r["label_index"])) for r in records])
    head_conf = np.array([float(r["confidence"]) for r in records])
    dist_conf = np.array([float(max(r["probs"])) for r in records])

    coverages = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    report = {
        "model": f"Verdict-2.0 ({backbone})",
        "benchmark": "LocalLLaMA/typed-decisions",
        "split": "test",
        "n_decisions": len(records),
        "base_accuracy": round(float(correct.mean()), 4),
        "auroc_correctness_head": round(auroc(head_conf, correct), 4),
        "auroc_distribution_maxp": round(auroc(dist_conf, correct), 4),
        "selective_correctness_head": selective_curve(head_conf, correct, coverages),
        "selective_distribution_maxp": selective_curve(dist_conf, correct, coverages),
        "note": "Quote auroc_correctness_head and the selective_correctness_head rows. "
                "The distribution rows are the stock max-p baseline for comparison.",
    }

    print(f"AUROC (correctness head) : {report['auroc_correctness_head']}")
    print(f"AUROC (max p baseline)   : {report['auroc_distribution_maxp']}")
    print(f"base accuracy            : {report['base_accuracy']}")
    for row in report["selective_correctness_head"]:
        if row["coverage"] in (0.8, 0.6):
            print(f"  coverage {row['coverage']:.0%} -> retained accuracy {row['retained_accuracy']:.4f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
