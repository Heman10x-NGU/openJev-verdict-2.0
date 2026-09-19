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


def permutation_stability(model, items, pad_id, device, batch_size, trials=5, seed=0):
    """Measure order sensitivity the way Kev reports it, so the numbers compare directly.

    Kev's released checkpoint records argmax_flip_rate 0.0741, mean_prob_spread 0.0653 and
    p90_prob_spread 0.2486 on this perturbation. Spread is the range of the probability assigned to
    the originally-top option across orderings, which catches models that keep their argmax while
    still moving a lot of mass.
    """
    import random as _random

    from .data import permuted_twin

    rng = _random.Random(seed)
    base = {}
    for r in infer(model, items, pad_id, device, batch_size, True):
        base[(r["case_id"], r["qid"])] = (int(np.argmax(r["probs"])), r["probs"])

    flips = total = 0
    observed = defaultdict(list)
    for _ in range(trials):
        twins, orders = [], {}
        for item in items:
            built = permuted_twin(TOKENIZER, item, rng)
            if built is None:
                continue
            twin, order = built
            twins.append(twin)
            orders[(twin.case_id, twin.qid)] = order
        if not twins:
            break
        for r in infer(model, twins, pad_id, device, batch_size, True):
            key = (r["case_id"], r["qid"])
            if key not in base or key not in orders:
                continue
            order = orders[key]
            probs = np.asarray(r["probs"], dtype=float)
            # map the twin distribution back onto the original option ordering
            aligned = np.zeros_like(probs)
            for position, original in enumerate(order):
                if position < len(probs) and original < len(aligned):
                    aligned[original] = probs[position]
            base_arg, base_probs = base[key]
            total += 1
            flips += int(int(np.argmax(aligned)) != base_arg)
            if base_arg < len(aligned):
                observed[key].append(float(aligned[base_arg]))

    spreads = [max(v) - min(v) for v in observed.values() if len(v) > 1]
    return {
        "argmax_flip_rate": round(float(flips / total), 4) if total else 0.0,
        "mean_prob_spread": round(float(np.mean(spreads)), 4) if spreads else 0.0,
        "p90_prob_spread": round(float(np.percentile(spreads, 90)), 4) if spreads else 0.0,
        "n_perturbed": total,
        "kev_reference": {"argmax_flip_rate": 0.0741, "mean_prob_spread": 0.0653, "p90_prob_spread": 0.2486},
    }


TOKENIZER = None  # set in main(); permuted_twin needs it to re-tokenize


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
    global TOKENIZER
    TOKENIZER = tokenizer

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
        "permutation_stability": permutation_stability(model, items, pad_id, device, args.batch_size),
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
