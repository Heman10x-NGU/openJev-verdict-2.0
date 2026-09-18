#!/usr/bin/env python3
"""Phase 0: determine whether Verdict 1.0's sub-random score is a defect or an architecture limit.

Verdict 1.0 scores 26.10% where uniform guessing scores about 26.9%, with a Brier of 0.5851 against
uniform's 0.2433. Being reliably worse than chance takes information, which is the signature of a
wiring defect rather than a weak model. This script runs four controls to locate it.

    python scripts/analysis/phase0_diagnose.py --limit 120

Reading the results:
  - shuffled_options accuracy materially different from baseline  -> option identity is misaligned
  - id_as_description accuracy materially higher than baseline    -> id and description are swapped
  - reversed_options accuracy materially higher than baseline     -> option order is inverted
  - all four within noise of each other and of uniform            -> the model contributes no signal
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def evaluate_variant(engine: Any, records: List[Dict], variant: str, seed: int = 0) -> Dict[str, float]:
    """Score the v1 engine under one perturbation of the option list."""
    from core.primitives import Choice, Level, Noul, Option, Query, Score

    rng = random.Random(seed)
    correct: List[float] = []
    confidences: List[float] = []

    for record in records:
        questions = record["questions"]
        gold = record["gold"]
        queries: List[Query] = []
        meta: Dict[str, Dict[str, Any]] = {}

        for qid, qdef in questions.items():
            qtype = qdef["type"]
            criteria = qdef.get("criteria")
            if qtype == "choice":
                pairs = list(criteria.items())
                if variant == "shuffled_options":
                    rng.shuffle(pairs)
                elif variant == "reversed_options":
                    pairs = pairs[::-1]
                if variant == "id_as_description":
                    options = [Option(id=k, description=k) for k, _ in pairs]
                else:
                    options = [Option(id=k, description=v or k) for k, v in pairs]
                queries.append(Choice(id=qid, question=qdef["instructions"], options=options))
            elif qtype == "noul":
                queries.append(Noul(id=qid, proposition=qdef["instructions"], semantics="conditional_on_sufficient_evidence_v2"))
            else:
                levels = [Level(id=str(i), value=float(i), description=str(c)) for i, c in enumerate(criteria)]
                queries.append(Score(id=qid, question=qdef["instructions"], levels=levels))
            meta[qid] = {"type": qtype, "gold": gold.get(qid, {})}

        try:
            batch = engine.evaluate(record["state"], queries)
        except Exception as exc:  # a crash here is itself a finding
            print(f"  warning: case {record.get('id')} failed under {variant}: {exc}", file=sys.stderr)
            continue

        for result in batch.results:
            info = meta[result.id]
            gold_label = str(info["gold"].get("label", ""))
            if info["type"] == "choice":
                predicted, confidence = result.selected_id, float(result.selected_probability)
            elif info["type"] == "noul":
                p_true = float(result.p_true_given_sufficient_evidence or 0.5)
                predicted = "true" if result.selected_outcome == "true" else "false"
                confidence = max(p_true, 1.0 - p_true)
                gold_label = gold_label.lower()
            else:
                predicted = result.selected_level_id
                confidence = float(result.probabilities.get(result.selected_level_id, 0.5))
            correct.append(float(predicted == gold_label))
            confidences.append(confidence)

    return {
        "n": len(correct),
        "accuracy": round(float(np.mean(correct)), 4) if correct else 0.0,
        "mean_confidence": round(float(np.mean(confidences)), 4) if confidences else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 0 diagnosis of the Verdict 1.0 baseline.")
    parser.add_argument("--model_path", default="artifacts/v2")
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=120, help="Test cases per variant. 120 gives about 1.5 points of noise.")
    parser.add_argument("--out", default="reports/phase0_diagnosis.json")
    args = parser.parse_args()

    import torch
    from datasets import load_dataset

    from core.engine_encoder import DecisionEngine

    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Loading Verdict 1.0 from {args.model_path} on {device}...")
    engine = DecisionEngine(model_name_or_path=args.model_path, device=device)

    ds = load_dataset("LocalLLaMA/typed-decisions", "all", split="test")
    records = []
    for i in range(min(args.limit, len(ds))):
        row = ds[i]
        records.append(
            {
                "id": row.get("id", ""),
                "state": row["state"],
                "questions": json.loads(row["questions"]) if isinstance(row["questions"], str) else row["questions"],
                "gold": json.loads(row["gold"]) if isinstance(row["gold"], str) else row["gold"],
            }
        )

    variants = ["baseline", "shuffled_options", "reversed_options", "id_as_description"]
    results: Dict[str, Dict[str, float]] = {}
    for variant in variants:
        print(f"\nRunning variant: {variant}")
        results[variant] = evaluate_variant(engine, records, variant)
        print(f"  accuracy {results[variant]['accuracy']:.4f}  mean confidence {results[variant]['mean_confidence']:.4f}")

    baseline = results["baseline"]["accuracy"]
    spread = max(v["accuracy"] for v in results.values()) - min(v["accuracy"] for v in results.values())
    verdict = (
        "WIRING DEFECT LIKELY: a perturbation that should not help changed accuracy materially."
        if spread > 0.05
        else "NO WIRING DEFECT FOUND: the engine is insensitive to option identity, so it contributes no signal."
    )

    report = {
        "baseline_accuracy": baseline,
        "variants": results,
        "accuracy_spread_across_variants": round(spread, 4),
        "uniform_random_reference": 0.269,
        "interpretation": verdict,
        "cases_per_variant": len(records),
    }
    print(f"\n{'=' * 72}\n{verdict}\nSpread across variants: {spread:.4f}\n{'=' * 72}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
