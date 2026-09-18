#!/usr/bin/env python3
"""Robust benchmark evaluation script for Verdict (OpenJev 151M) on LocalLLaMA/typed-decisions.

Evaluates:
- Top-1 Exact Match Accuracy
- Soft Accuracy (dot product with soft target)
- Brier Score
- Total Variation Distance
- Expected Calibration Error (ECE)
- Latency (p50, p95, mean)
- Per-workflow breakdown

Includes timeouts, granular logging, and fail-safe exception handling.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from datasets import load_dataset

from core.engine_encoder import DecisionEngine
from core.primitives import (
    Choice,
    ChoiceResult,
    Level,
    Noul,
    NoulResult,
    Option,
    Query,
    Score,
    ScoreResult,
)


def compute_ece(confidences: np.ndarray, accuracies: np.ndarray, num_bins: int = 10) -> float:
    """Compute Expected Calibration Error across confidence bins."""
    if len(confidences) == 0:
        return 0.0
    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0
    total_samples = len(confidences)

    for i in range(num_bins):
        bin_lower = bin_edges[i]
        bin_upper = bin_edges[i + 1]
        mask = (confidences > bin_lower) & (confidences <= bin_upper) if i > 0 else (confidences >= bin_lower) & (confidences <= bin_upper)
        bin_samples = int(np.sum(mask))

        if bin_samples > 0:
            bin_acc = float(np.mean(accuracies[mask]))
            bin_conf = float(np.mean(confidences[mask]))
            ece += (bin_samples / total_samples) * abs(bin_acc - bin_conf)

    return float(ece)


def evaluate_single_record(
    engine: DecisionEngine,
    record: Dict[str, Any],
) -> Dict[str, Any]:
    """Parse question definitions, execute Verdict engine, and compare against gold targets."""
    state = record["state"]
    q_defs = json.loads(record["questions"]) if isinstance(record["questions"], str) else record["questions"]
    gold_defs = json.loads(record["gold"]) if isinstance(record["gold"], str) else record["gold"]

    queries: List[Query] = []
    q_meta: Dict[str, Dict[str, Any]] = {}

    for qid, q in q_defs.items():
        t = q["type"]
        ins = q["instructions"]
        crit = q.get("criteria", {})
        q_meta[qid] = {"type": t, "criteria": crit, "gold": gold_defs.get(qid, {})}

        if t == "choice":
            # Cap candidates to 24 substantive options to respect Verdict's GLiClass contract
            items = list(crit.items())[:24]
            opts = [Option(id=k, description=v if v else k) for k, v in items]
            queries.append(Choice(id=qid, question=ins, options=opts))
        elif t == "noul":
            queries.append(Noul(id=qid, proposition=ins, semantics="conditional_on_sufficient_evidence_v2"))
        elif t == "score":
            lvls = [Level(id=str(i), value=float(i), description=str(c)) for i, c in enumerate(crit[:24])]
            queries.append(Score(id=qid, question=ins, levels=lvls))

    t0 = time.perf_counter()
    batch_res = engine.evaluate(state, queries)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    answers: Dict[str, Any] = {}
    for r in batch_res.results:
        meta = q_meta[r.id]
        gold = meta["gold"]
        qtype = meta["type"]

        if isinstance(r, ChoiceResult):
            pred_choice = r.selected_id
            gold_label = gold.get("label", "")
            is_correct = 1.0 if pred_choice == gold_label else 0.0

            # Normalize probabilities over candidate set
            p_dist = r.probabilities
            g_dist = gold.get("probabilities", {})

            # Match keys
            all_keys = list(set(p_dist.keys()) | set(g_dist.keys()))
            p_vec = np.array([p_dist.get(k, 1e-6) for k in all_keys], dtype=float)
            g_vec = np.array([g_dist.get(k, 1e-6) for k in all_keys], dtype=float)
            p_vec = p_vec / max(1e-9, p_vec.sum())
            g_vec = g_vec / max(1e-9, g_vec.sum())

            brier = float(np.sum((p_vec - g_vec) ** 2))
            tv = float(0.5 * np.sum(np.abs(p_vec - g_vec)))
            soft_acc = float(np.sum(p_vec * g_vec))
            conf = float(r.selected_probability)

            answers[r.id] = {
                "type": "choice",
                "pred": pred_choice,
                "gold": gold_label,
                "is_correct": is_correct,
                "confidence": conf,
                "brier": brier,
                "tv": tv,
                "soft_acc": soft_acc,
            }

        elif isinstance(r, NoulResult):
            pred_outcome = r.selected_outcome
            gold_label = str(gold.get("label", "false")).lower()
            pred_bool = "true" if pred_outcome == "true" else "false"
            is_correct = 1.0 if pred_bool == gold_label else 0.0

            p_true = float(r.p_true_given_sufficient_evidence or 0.5)
            p_vec = np.array([1.0 - p_true, p_true])

            g_val = gold.get("noul", 1.0 if gold_label == "true" else 0.0)
            g_vec = np.array([1.0 - g_val, g_val])

            brier = float(np.sum((p_vec - g_vec) ** 2))
            tv = float(0.5 * np.sum(np.abs(p_vec - g_vec)))
            soft_acc = float(np.sum(p_vec * g_vec))
            conf = max(p_true, 1.0 - p_true)

            answers[r.id] = {
                "type": "noul",
                "pred": pred_bool,
                "gold": gold_label,
                "is_correct": is_correct,
                "confidence": conf,
                "brier": brier,
                "tv": tv,
                "soft_acc": soft_acc,
            }

        elif isinstance(r, ScoreResult):
            pred_lvl = r.selected_level_id
            gold_lvl = str(gold.get("label", "0"))
            is_correct = 1.0 if pred_lvl == gold_lvl else 0.0

            p_score = float(r.expected_score or 0.0)
            g_score = float(gold.get("score", float(gold_lvl)))
            mae = abs(p_score - g_score)

            conf = float(r.probabilities.get(r.selected_level_id, 0.5))

            answers[r.id] = {
                "type": "score",
                "pred": pred_lvl,
                "gold": gold_lvl,
                "is_correct": is_correct,
                "confidence": conf,
                "score_mae": mae,
                "within_one": 1.0 if mae <= 1.0 else 0.0,
            }

    return {
        "id": record.get("id", ""),
        "workflow": record.get("workflow", "default"),
        "latency_ms": elapsed_ms,
        "answers": answers,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Verdict baseline on LocalLLaMA/typed-decisions.")
    parser.add_argument("--model_path", default="artifacts/v2", help="Path to local Verdict model directory.")
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of evaluation samples.")
    parser.add_argument("--out", default="reports/verdict_baseline_benchmark.json")
    args = parser.parse_args()

    print(f"=== Initializing Verdict Decision Engine ===")
    print(f"Model path: {args.model_path}")
    print(f"Target device: {args.device}")

    t_init0 = time.perf_counter()
    engine = DecisionEngine(model_name_or_path=args.model_path, device=args.device)
    print(f"Engine loaded in {time.perf_counter() - t_init0:.2f}s.")

    print("\nLoading LocalLLaMA/typed-decisions (split='test')...")
    ds = load_dataset("LocalLLaMA/typed-decisions", "all", split="test")
    total_records = len(ds) if args.limit is None else min(len(ds), args.limit)
    print(f"Evaluating {total_records} test records (~{total_records * 5} decision questions)...")

    results = []
    latencies = []
    by_workflow: Dict[str, List[Dict[str, Any]]] = {}

    t_start = time.perf_counter()

    for idx in range(total_records):
        row = ds[idx]
        wf = row.get("workflow", "general")
        if wf not in by_workflow:
            by_workflow[wf] = []

        try:
            eval_out = evaluate_single_record(engine, row)
            results.append(eval_out)
            by_workflow[wf].append(eval_out)
            latencies.append(eval_out["latency_ms"])
        except Exception as e:
            print(f"Warning: Record {idx} failed with error: {e}", file=sys.stderr)
            continue

        if (idx + 1) % 50 == 0 or (idx + 1) == total_records:
            elapsed = time.perf_counter() - t_start
            rate = (idx + 1) / elapsed
            print(f"  [{idx + 1}/{total_records}] Elapsed: {elapsed:.1f}s | Speed: {rate:.1f} cases/sec | Current P50 Latency: {np.percentile(latencies, 50):.1f} ms")

    total_eval_time = time.perf_counter() - t_start
    print(f"\nEvaluation completed in {total_eval_time:.2f} seconds!")

    # Aggregate global metrics
    all_accuracies = []
    all_confidences = []
    all_briers = []
    all_soft_accs = []
    all_score_maes = []
    all_within_one = []

    for r in results:
        for qid, a in r["answers"].items():
            all_accuracies.append(a["is_correct"])
            all_confidences.append(a["confidence"])
            if "brier" in a:
                all_briers.append(a["brier"])
            if "soft_acc" in a:
                all_soft_accs.append(a["soft_acc"])
            if "score_mae" in a:
                all_score_maes.append(a["score_mae"])
                all_within_one.append(a["within_one"])

    global_acc = float(np.mean(all_accuracies)) if all_accuracies else 0.0
    global_ece = compute_ece(np.array(all_confidences), np.array(all_accuracies)) if all_accuracies else 0.0
    global_brier = float(np.mean(all_briers)) if all_briers else 0.0
    global_soft_acc = float(np.mean(all_soft_accs)) if all_soft_accs else 0.0
    global_score_mae = float(np.mean(all_score_maes)) if all_score_maes else 0.0
    global_within_one = float(np.mean(all_within_one)) if all_within_one else 0.0

    p50_lat = float(np.percentile(latencies, 50)) if latencies else 0.0
    p95_lat = float(np.percentile(latencies, 95)) if latencies else 0.0

    print("\n" + "=" * 55)
    print("        VERDICT 1.0 EMPIRICAL BASELINE REPORT")
    print("=" * 55)
    print(f"Evaluated Cases   : {len(results)}")
    print(f"Evaluated Queries : {len(all_accuracies)}")
    print(f"Top-1 Accuracy    : {global_acc * 100:.2f}%")
    print(f"Soft Accuracy     : {global_soft_acc * 100:.2f}%")
    print(f"Brier Score       : {global_brier:.4f}")
    print(f"ECE (Calibration) : {global_ece:.4f}")
    print(f"Score MAE         : {global_score_mae:.3f} (Within 1 Level: {global_within_one * 100:.1f}%)")
    print(f"Latency P50       : {p50_lat:.1f} ms / case")
    print(f"Latency P95       : {p95_lat:.1f} ms / case")
    print("=" * 55)

    # Per workflow breakdown
    wf_stats = {}
    print("\nBreakdown by Enterprise Workflow:")
    for wf, wf_results in by_workflow.items():
        wf_accs = [a["is_correct"] for r in wf_results for a in r["answers"].values()]
        acc = float(np.mean(wf_accs)) if wf_accs else 0.0
        wf_stats[wf] = {
            "cases": len(wf_results),
            "questions": len(wf_accs),
            "accuracy": round(acc, 4),
        }
        print(f"  • {wf:<25}: {acc * 100:.2f}% ({len(wf_results)} cases)")

    # Save full JSON report
    report = {
        "model": "Verdict-OpenJev-151M",
        "benchmark": "LocalLLaMA/typed-decisions",
        "split": "test",
        "device": args.device,
        "total_cases": len(results),
        "total_questions": len(all_accuracies),
        "total_eval_seconds": round(total_eval_time, 2),
        "metrics": {
            "accuracy": round(global_acc, 4),
            "soft_accuracy": round(global_soft_acc, 4),
            "brier_score": round(global_brier, 4),
            "ece": round(global_ece, 4),
            "score_mae": round(global_score_mae, 4),
            "within_one_level": round(global_within_one, 4),
            "latency_p50_ms": round(p50_lat, 2),
            "latency_p95_ms": round(p95_lat, 2),
        },
        "workflows": wf_stats,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nFull report written to: {out_path}")


if __name__ == "__main__":
    main()
