#!/usr/bin/env python3
"""Single final read of the test vault. Run this once per candidate model, after everything is frozen.

    python -m verdict2.evaluate --checkpoint artifacts/verdict2/model.pt --out reports/verdict2_test.json
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from .data import load_items
from .metrics import summarize
from .model import VerdictModel
from .train import batches, infer, pick_device

CHECKLIST = [
    "Temperature was fit on the calib fold, not test.",
    "The correctness head was trained on calib-fold predictions, not test.",
    "Early stopping and every hyperparameter choice used the dev fold, not test.",
    "This is a deliberate, counted read of the test vault.",
]


def permutation_flip_rate(model, items, pad_id, device, batch_size, trials=5, seed=0):
    """Fraction of choice questions whose argmax changes when options are reordered."""
    from .data import build_item  # noqa: F401  (documented dependency)

    base = {(r["case_id"], r["qid"]): int(np.argmax(r["probs"])) for r in infer(model, items, pad_id, device, batch_size, True)}
    rng = random.Random(seed)
    flips = total = 0
    for _ in range(trials):
        shuffled = []
        for it in items:
            width = len(it.markers)
            if it.qtype != 0 or width < 3:
                continue
            order = list(range(width))
            rng.shuffle(order)
            shuffled.append((it, order))
        if not shuffled:
            break
        subset = [it for it, _ in shuffled]
        for r in infer(model, subset, pad_id, device, batch_size, True):
            key = (r["case_id"], r["qid"])
            if key in base:
                total += 1
                flips += int(int(np.argmax(r["probs"])) != base[key])
    return float(flips / total) if total else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Final test-set evaluation for Verdict 2.0.")
    parser.add_argument("--checkpoint", default="artifacts/verdict2/model.pt")
    parser.add_argument("--out", default="reports/verdict2_test.json")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=None, help="Cap test cases. Smoke tests only; a real read uses all 400.")
    parser.add_argument("--confirm", action="store_true", help="Acknowledge the anti-leak checklist.")
    args = parser.parse_args()

    if not args.confirm:
        print("Anti-leak checklist. Confirm each item, then rerun with --confirm:")
        for line in CHECKLIST:
            print(f"  [ ] {line}")
        raise SystemExit(1)

    device = pick_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    backbone = checkpoint["backbone"]

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(backbone)
    pad_id = tokenizer.pad_token_id or 0

    model = VerdictModel(backbone).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    items = load_items("test", tokenizer, limit=args.limit)
    if args.limit:
        print(f"WARNING: --limit {args.limit} is a smoke test, not a reportable result.")
    print(f"Evaluating {len(items)} test decisions on {device}...")

    start = time.perf_counter()
    records = infer(model, items, pad_id, device, args.batch_size, use_temperature=True)
    elapsed = time.perf_counter() - start

    metrics = summarize(records)
    by_workflow = defaultdict(list)
    for r in records:
        by_workflow[r["workflow"]].append(r)

    report = {
        "model": f"Verdict-2.0 ({backbone})",
        "benchmark": "LocalLLaMA/typed-decisions",
        "split": "test",
        "device": device,
        "total_decisions": len(records),
        "metrics": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in metrics.items()},
        "permutation_argmax_flip_rate": round(permutation_flip_rate(model, items, pad_id, device, args.batch_size), 4),
        "throughput_decisions_per_second": round(len(records) / max(elapsed, 1e-9), 1),
        "workflows": {
            wf: {"decisions": len(rows), "accuracy": round(summarize(rows)["accuracy"], 4)}
            for wf, rows in sorted(by_workflow.items())
        },
        "reference_floors_note": "Compare against reports/reference_floors.json in the same report.",
    }

    print(json.dumps(report["metrics"], indent=2))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
