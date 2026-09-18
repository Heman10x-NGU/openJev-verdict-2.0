"""Stage 3: Parallel metric analysis and receipt generation for Gen Suite.

Reads cached .npz logit files, computes task-subset accuracy, overall accuracy, Brier score,
NLL, ECE (adaptive, equal-width, equal-mass), MCE, abstention recall/precision/F1,
over-abstention rate, selective-risk curves, and 1000-sample bootstrap CIs. Emits per-task
receipts and the master summary matrix in reports/v2/gen_sweep_summary.json with promotion rankings.
"""

from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import sys
from pathlib import Path
from typing import Any

WORKSPACE_DIR = Path(__file__).resolve().parent.parent.parent
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

import numpy as np

from core.calibration import compute_bootstrap_ci, compute_ece
from scripts.gen_suite.tasks import INSUFFICIENT_EVIDENCE_ID, TASK_REGISTRY, TaskSpec

CACHE_DIR = WORKSPACE_DIR / "reports" / "v2" / "_cache"
REPORTS_DIR = WORKSPACE_DIR / "reports" / "v2"


def analyze_npz(npz_path: Path) -> dict[str, Any]:
    """Load one cached .npz file and compute all empirical calibration metrics."""
    data = np.load(npz_path, allow_pickle=True)
    logits = data["logits"].astype(np.float64)  # Shape: (N, K)
    targets = data["target_idx"].astype(np.int64)  # Shape: (N,)
    item_ids = data["item_ids"].tolist()
    candidate_ids = data["candidate_ids"].tolist()
    task_id = str(data["task_id"])
    checkpoint = str(data["checkpoint"])
    arm = str(data["arm"])

    num_samples = len(targets)
    k_classes = logits.shape[1]

    # Numerically stable softmax
    max_logits = np.max(logits, axis=-1, keepdims=True)
    exp_logits = np.exp(logits - max_logits)
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

    preds = np.argmax(probs, axis=-1)
    confs = np.max(probs, axis=-1)

    corrects = (preds == targets).astype(np.float64)
    accuracy_overall = float(np.mean(corrects))

    # Abstention metrics
    abstain_idx = (
        candidate_ids.index(INSUFFICIENT_EVIDENCE_ID)
        if INSUFFICIENT_EVIDENCE_ID in candidate_ids
        else -1
    )
    is_abstain_pred = (
        (preds == abstain_idx) if abstain_idx >= 0 else np.zeros(num_samples, dtype=bool)
    )
    is_abstain_gold = (
        (targets == abstain_idx) if abstain_idx >= 0 else np.zeros(num_samples, dtype=bool)
    )

    in_scope_mask = ~is_abstain_gold
    in_scope_count = int(np.sum(in_scope_mask))
    abstain_gold_count = int(np.sum(is_abstain_gold))

    # Task-subset accuracy: evaluated strictly on in-scope items
    if in_scope_count > 0:
        accuracy_task_subset = float(np.mean(corrects[in_scope_mask]))
    else:
        accuracy_task_subset = accuracy_overall

    # Confusion counts for abstention
    tp = int(np.sum(is_abstain_pred & is_abstain_gold))
    fp = int(np.sum(is_abstain_pred & ~is_abstain_gold))
    fn = int(np.sum(~is_abstain_pred & is_abstain_gold))
    tn = int(np.sum(~is_abstain_pred & ~is_abstain_gold))

    abs_precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    abs_recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    abs_f1 = (
        float(2 * abs_precision * abs_recall / (abs_precision + abs_recall))
        if (abs_precision + abs_recall) > 0
        else 0.0
    )

    abstention_rate = float(np.mean(is_abstain_pred))
    over_abstention_rate = float(fp / in_scope_count) if in_scope_count > 0 else 0.0

    # Strictly proper scoring rules: NLL and Multiclass Brier score
    eps = 1e-15
    clipped_probs = np.clip(probs, eps, 1.0 - eps)
    target_probs = clipped_probs[np.arange(num_samples), targets]
    nll = float(-np.mean(np.log(target_probs)))

    one_hot = np.zeros_like(probs)
    one_hot[np.arange(num_samples), targets] = 1.0
    brier_score = float(np.mean(np.sum((probs - one_hot) ** 2, axis=-1)))

    # ECE & MCE
    ece_ew = compute_ece(confs, corrects, n_bins=10, strategy="equal_width", min_bin_count=10)
    ece_em = compute_ece(confs, corrects, n_bins=10, strategy="equal_mass", min_bin_count=10)
    ece_ad = compute_ece(confs, corrects, n_bins=10, strategy="adaptive", min_bin_count=10)

    # 1000-sample bootstrap 95% CIs
    indices = np.arange(num_samples)
    ci_acc_overall = compute_bootstrap_ci(
        indices, lambda idxs: float(np.mean(corrects[idxs])), n_resamples=1000
    )
    if in_scope_count > 0:
        in_scope_indices = np.where(in_scope_mask)[0]
        ci_acc_subset = compute_bootstrap_ci(
            in_scope_indices,
            lambda idxs: float(np.mean(corrects[idxs])),
            n_resamples=1000,
        )
        ci_over_abs = compute_bootstrap_ci(
            in_scope_indices,
            lambda idxs: float(np.mean(is_abstain_pred[idxs])),
            n_resamples=1000,
        )
    else:
        ci_acc_subset = ci_acc_overall
        ci_over_abs = (0.0, 0.0)

    ci_ece_ad = compute_bootstrap_ci(
        indices,
        lambda idxs: compute_ece(
            confs[idxs], corrects[idxs], n_bins=10, strategy="adaptive", min_bin_count=10
        ).ece,
        n_resamples=1000,
    )
    ci_ece_ew = compute_bootstrap_ci(
        indices,
        lambda idxs: compute_ece(
            confs[idxs], corrects[idxs], n_bins=10, strategy="equal_width", min_bin_count=10
        ).ece,
        n_resamples=1000,
    )
    ci_abs_rate = compute_bootstrap_ci(
        indices, lambda idxs: float(np.mean(is_abstain_pred[idxs])), n_resamples=1000
    )

    # Selective-risk curve, thresholds 0.50 to 0.99 step 0.01
    thresholds = [round(t, 2) for t in np.arange(0.50, 1.00, 0.01)]
    selective_curve = []
    for tau in thresholds:
        accepted = confs >= tau
        accepted_cnt = int(np.sum(accepted))
        coverage = float(accepted_cnt / num_samples) if num_samples > 0 else 0.0
        if accepted_cnt > 0:
            errors_cnt = int(np.sum(corrects[accepted] == 0))
            risk = float(errors_cnt / accepted_cnt)
            retained_acc = float(1.0 - risk)
            z = 1.96
            denom = 1 + (z**2) / accepted_cnt
            centre = (risk + (z**2) / (2 * accepted_cnt)) / denom
            spread = (
                z
                * math.sqrt((risk * (1 - risk) + (z**2) / (4 * accepted_cnt)) / accepted_cnt)
                / denom
            )
            risk_ub_95 = float(min(1.0, centre + spread))
        else:
            errors_cnt = 0
            risk = 0.0
            retained_acc = 1.0
            risk_ub_95 = 0.0

        selective_curve.append(
            {
                "threshold": tau,
                "coverage": coverage,
                "retained_accuracy": retained_acc,
                "accepted_count": accepted_cnt,
                "errors": errors_cnt,
                "selective_risk": risk,
                "risk_upper_bound_95": risk_ub_95,
            }
        )

    return {
        "task_id": task_id,
        "checkpoint": checkpoint,
        "arm": arm,
        "num_samples": num_samples,
        "num_in_scope": in_scope_count,
        "num_abstain_gold": abstain_gold_count,
        "k_cardinality": k_classes,
        "accuracy_task_subset": accuracy_task_subset,
        "accuracy_overall": accuracy_overall,
        "negative_log_likelihood": nll,
        "brier_score": brier_score,
        "ece_adaptive": ece_ad.ece,
        "mce_adaptive": ece_ad.mce,
        "ece_equal_width": ece_ew.ece,
        "mce_equal_width": ece_ew.mce,
        "ece_equal_mass": ece_em.ece,
        "mce_equal_mass": ece_em.mce,
        "abstention_rate": abstention_rate,
        "over_abstention_rate": over_abstention_rate,
        "abstention": {
            "precision": abs_precision,
            "recall": abs_recall,
            "f1_score": abs_f1,
            "true_abstentions": tp,
            "false_abstentions": fp,
            "false_negatives": fn,
            "true_negatives": tn,
            "total_gold_abstentions": abstain_gold_count,
            "total_in_scope": in_scope_count,
        },
        "ci_95": {
            "accuracy_task_subset": list(ci_acc_subset),
            "accuracy_overall": list(ci_acc_overall),
            "ece_adaptive": list(ci_ece_ad),
            "ece_equal_width": list(ci_ece_ew),
            "over_abstention_rate": list(ci_over_abs),
            "abstention_rate": list(ci_abs_rate),
        },
        "selective_policy": selective_curve,
    }


def evaluate_promotion_gates(
    task_spec: TaskSpec, task_results: list[dict[str, Any]], is_smoke: bool = False
) -> dict[str, Any]:
    """Evaluate promotion criteria across all 6 configurations for a task.
    
    Ranking is strictly by accuracy_task_subset (with tie-breakers on lowest adaptive ECE and lowest Brier).
    Only tasks with provenance == 'public' can pass promotion gates.
    """
    min_items = 20 if is_smoke else 500

    # Sort configurations strictly by accuracy_task_subset descending
    ranked_configs = sorted(
        task_results,
        key=lambda r: (
            r["accuracy_task_subset"],
            -r["ece_adaptive"],
            -r["brier_score"],
        ),
        reverse=True,
    )

    selected_best = ranked_configs[0]

    # Evaluate gates on best candidate
    g1_provenance = task_spec.provenance == "public"
    g2_acc = selected_best["accuracy_task_subset"] >= 0.85
    g3_ece = selected_best["ece_adaptive"] <= 0.05
    g4_over_abs = selected_best["over_abstention_rate"] <= 0.10
    g5_count = selected_best["num_samples"] >= min_items
    g6_legible = True

    passed_all_gates = bool(
        g1_provenance and g2_acc and g3_ece and g4_over_abs and g5_count and g6_legible
    )

    gate_status = "PASSED" if passed_all_gates else "FAILED"

    failure_reasons = []
    if not g1_provenance:
        failure_reasons.append(f"Provenance is '{task_spec.provenance}' (authored tasks barred from promotion)")
    if not g2_acc:
        failure_reasons.append(f"Task-subset accuracy {selected_best['accuracy_task_subset']*100:.1f}% < 85.0%")
    if not g3_ece:
        failure_reasons.append(f"Adaptive ECE {selected_best['ece_adaptive']*100:.2f}% > 5.0%")
    if not g4_over_abs:
        failure_reasons.append(f"Over-abstention rate {selected_best['over_abstention_rate']*100:.1f}% > 10.0%")
    if not g5_count:
        failure_reasons.append(f"Sample count {selected_best['num_samples']} < {min_items}")

    return {
        "passed_all_gates": passed_all_gates,
        "gate_status": gate_status,
        "failure_reasons": failure_reasons,
        "selected_max_of_6": {
            "checkpoint": selected_best["checkpoint"],
            "arm": selected_best["arm"],
            "accuracy_task_subset": selected_best["accuracy_task_subset"],
            "accuracy_overall": selected_best["accuracy_overall"],
            "ece_adaptive": selected_best["ece_adaptive"],
            "over_abstention_rate": selected_best["over_abstention_rate"],
            "brier_score": selected_best["brier_score"],
            "negative_log_likelihood": selected_best["negative_log_likelihood"],
            "num_samples": selected_best["num_samples"],
        },
        "gates": {
            "public_provenance": g1_provenance,
            "accuracy_task_subset_ge_85": g2_acc,
            "ece_adaptive_le_5": g3_ece,
            "over_abstention_le_10": g4_over_abs,
            "sample_count_ge_min": g5_count,
            "single_sentence_legibility": g6_legible,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Stage 3: Parallel metric analysis and receipt generation.")
    parser.add_argument("--smoke", action="store_true", help="Running on smoke caches.")
    parser.add_argument("--workers", type=int, default=8, help="Number of analysis worker processes.")
    args = parser.parse_args()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    npz_files = sorted(CACHE_DIR.glob("logits_*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No cached logits found in {CACHE_DIR}. Run run_inference.py first.")

    print(f"=== Stage 3: Analyzing {len(npz_files)} Logit Caches across 12 Tasks ===")

    results_by_task: dict[str, list[dict[str, Any]]] = {}

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(analyze_npz, f): f for f in npz_files}
        for future in as_completed(futures):
            res = future.result()
            task_id = res["task_id"]
            results_by_task.setdefault(task_id, []).append(res)

    # Save per-task receipts and master summary
    task_summaries: list[dict[str, Any]] = []

    for task_id, task_results in sorted(results_by_task.items()):
        spec = TASK_REGISTRY[task_id]
        promo = evaluate_promotion_gates(spec, task_results, is_smoke=args.smoke)

        task_receipt = {
            "task_id": task_id,
            "name": spec.name,
            "cookbook": spec.cookbook,
            "source": spec.source,
            "provenance": spec.provenance,
            "provenance_details": spec.provenance_details,
            "k_cardinality": spec.k_cardinality,
            "demo_value": spec.demo_value,
            "description": spec.description,
            "promotion_evaluation": promo,
            "matrix": sorted(task_results, key=lambda x: (x["checkpoint"], x["arm"])),
        }

        receipt_file = REPORTS_DIR / f"receipt_{task_id.lower()}.json"
        with open(receipt_file, "w", encoding="utf-8") as f:
            json.dump(task_receipt, f, indent=2)

        task_summaries.append(task_receipt)

    # Sort tasks by:
    # 1. Gate passed status (public and passed all gates)
    # 2. Highest accuracy_task_subset
    task_summaries.sort(
        key=lambda t: (
            t["promotion_evaluation"]["passed_all_gates"],
            t["promotion_evaluation"]["selected_max_of_6"]["accuracy_task_subset"],
        ),
        reverse=True,
    )

    promoted_task_ids = [
        t["task_id"] for t in task_summaries if t["promotion_evaluation"]["passed_all_gates"]
    ]

    master_summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "is_smoke": bool(args.smoke),
        "total_tasks_evaluated": len(task_summaries),
        "promoted_tasks_count": len(promoted_task_ids),
        "promoted_tasks": promoted_task_ids,
        "ranked_tasks": task_summaries,
    }

    summary_file = REPORTS_DIR / "gen_sweep_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(master_summary, f, indent=2)

    print(f"\nSaved master sweep summary to {summary_file}")

    # Print clean terminal matrix and promotion leaderboard
    print("\n================================================= GEN SUITE 12-TASK BENCHMARK MATRIX =================================================")
    print(f"{'Task ID':<7} | {'Task Name':<30} | {'Prov':<8} | {'Ckpt':<9} | {'Arm':<14} | {'Task Acc':<8} | {'All Acc':<7} | {'ECE (Ad)':<8} | {'Over-Abs':<8} | {'Brier':<6}")
    print("-" * 130)

    for task_rec in sorted(task_summaries, key=lambda x: x["task_id"]):
        for r in sorted(task_rec["matrix"], key=lambda x: (x["checkpoint"], x["arm"])):
            print(
                f"{r['task_id']:<7} | {task_rec['name']:<30} | {task_rec['provenance']:<8} | {r['checkpoint']:<9} | {r['arm']:<14} | "
                f"{r['accuracy_task_subset']*100:6.1f}% | {r['accuracy_overall']*100:5.1f}% | {r['ece_adaptive']*100:6.2f}% | {r['over_abstention_rate']*100:6.1f}% | {r['brier_score']:6.3f}"
            )

    print("\n=================================================== RANKED PROMOTION LEADERBOARD ===================================================")
    print(f"{'Rank':<4} | {'Task ID':<7} | {'Task Name':<30} | {'Prov':<8} | {'Winning Ckpt':<12} | {'Winning Arm':<14} | {'Task Acc':<8} | {'ECE (Ad)':<8} | {'Gate Status'}")
    print("-" * 130)

    for rank_idx, t in enumerate(task_summaries, start=1):
        pe = t["promotion_evaluation"]
        winner = pe["selected_max_of_6"]
        status = "PASSED (PROMOTED)" if pe["passed_all_gates"] else f"FAILED ({', '.join(pe['failure_reasons'][:1])})"
        print(
            f"{rank_idx:<4} | {t['task_id']:<7} | {t['name']:<30} | {t['provenance']:<8} | {winner['checkpoint']:<12} | {winner['arm']:<14} | "
            f"{winner['accuracy_task_subset']*100:6.1f}% | {winner['ece_adaptive']*100:6.2f}% | {status}"
        )
    print("====================================================================================================================================")


if __name__ == "__main__":
    main()

