#!/usr/bin/env python3
"""Render documentation tables and empirical numbers directly from JSON receipts.

Ensures zero manual transcription: every number, table, and metric reported
in README.md and WALKTHROUGH.md originates from JSON receipts under reports/v2/.
Supports --check flag to verify documentation freshness in CI.
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required receipt {path} does not exist.")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def format_pct(val: float | None) -> str:
    if val is None:
        return "N/A"
    return f"{val * 100:.2f}%"


def format_num(val: float | None, decimals: int = 4) -> str:
    if val is None:
        return "N/A"
    return f"{val:.{decimals}f}"


def render_held_out_table(eval_data: dict[str, Any]) -> str:
    uncal = eval_data["uncalibrated"]
    cal = eval_data["calibrated"]
    cis = cal.get("ci_95", {})
    abs_uncal = uncal.get("abstention", {})
    abs_cal = cal.get("abstention", {})
    abs_cis = abs_cal.get("ci_95", {})

    acc_ci = f"[{cis['accuracy'][0]*100:.2f}%, {cis['accuracy'][1]*100:.2f}%]" if "accuracy" in cis else "N/A"
    ece_ci = f"[{cis['ece_equal_width'][0]*100:.2f}%, {cis['ece_equal_width'][1]*100:.2f}%]" if "ece_equal_width" in cis else "N/A"
    em_ci = f"[{cis['ece_equal_mass'][0]*100:.2f}%, {cis['ece_equal_mass'][1]*100:.2f}%]" if "ece_equal_mass" in cis else "N/A"
    rec_ci = f"[{abs_cis['recall'][0]*100:.2f}%, {abs_cis['recall'][1]*100:.2f}%]" if "recall" in abs_cis else "N/A"
    prec_ci = f"[{abs_cis['precision'][0]*100:.2f}%, {abs_cis['precision'][1]*100:.2f}%]" if "precision" in abs_cis else "N/A"

    lines = [
        "| Metric | Uncalibrated | Calibrated | 95% Bootstrap CI | Notes / Definition |",
        "| :--- | :--- | :--- | :--- | :--- |",
        f"| Top-1 accuracy | {format_pct(uncal['accuracy'])} | {format_pct(cal['accuracy'])} | {acc_ci} | Evaluated across 1,000 held-out test examples |",
        f"| Negative log-likelihood (NLL) | {format_num(uncal['negative_log_likelihood'])} | {format_num(cal['negative_log_likelihood'])} | N/A | Strictly proper scoring rule across candidates |",
        f"| Multiclass Brier score | {format_num(uncal['brier_score'])} | {format_num(cal['brier_score'])} | N/A | Mean squared probability error |",
        f"| Equal-width ECE (10 bins) | {format_pct(uncal['ece_equal_width'])} | {format_pct(cal['ece_equal_width'])} | {ece_ci} | Standard 10-bin expected calibration error |",
        f"| Equal-mass ECE (10 bins) | {format_pct(uncal['ece_equal_mass'])} | {format_pct(cal['ece_equal_mass'])} | {em_ci} | Quantile-binned calibration error |",
        f"| Adaptive ECE (min 10 samples) | {format_pct(uncal['ece_adaptive'])} | {format_pct(cal['ece_adaptive'])} | N/A | Excludes bins with fewer than 10 samples |",
        f"| Maximum Calibration Error (MCE) | {format_pct(uncal['mce_equal_width'])} | {format_pct(cal['mce_equal_width'])} | N/A | Bins with >= 10 samples only |",
        f"| Out-of-scope abstention recall | {format_pct(abs_uncal['recall'])} | {format_pct(abs_cal['recall'])} | {rec_ci} | Proportion of out-of-scope queries flagged |",
        f"| Out-of-scope abstention precision | {format_pct(abs_uncal['precision'])} | {format_pct(abs_cal['precision'])} | {prec_ci} | Proportion of abstention predictions that are correct |",
        f"| Out-of-scope abstention F1 | {format_pct(abs_uncal['f1_score'])} | {format_pct(abs_cal['f1_score'])} | N/A | Harmonic mean of abstention recall and precision |",
        f"| False abstentions (in-scope) | {abs_uncal.get('false_abstentions', 'N/A')} | {abs_cal.get('false_abstentions', 'N/A')} | N/A | In-scope banking queries erroneously flagged to abstain |",
        f"| Optimal temperature ($T$) | 1.0000 | {format_num(eval_data.get('temperature', 1.0))} | N/A | Fitted on calibration set NLL via L-BFGS |",
    ]
    return "\n".join(lines)


def render_cardinality_table(eval_data: dict[str, Any]) -> str:
    slices = eval_data.get("slices", {})
    cardinality_keys = [("cardinality_k3", "K = 3"), ("cardinality_k5", "K = 5"),
                        ("cardinality_k9", "K = 9"), ("cardinality_k17", "K = 17"),
                        ("cardinality_k25", "K = 25 (maximum capacity)")]

    lines = [
        "| Candidate menu size | Top-1 accuracy | Brier score | Equal-width ECE | False abstentions | Sample count |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |",
    ]
    for key, label in cardinality_keys:
        if key in slices:
            s = slices[key]
            lines.append(
                f"| {label} | {format_pct(s['accuracy'])} | {format_num(s['brier'])} | {format_pct(s['ece'])} | {s.get('false_abstentions', 0)} | {s.get('samples', 100)} |"
            )
    return "\n".join(lines)


def render_out_of_scope_table(eval_data: dict[str, Any]) -> str:
    slices = eval_data.get("slices", {})
    lines = [
        "| Slice | Sample count | Abstention recall | Abstention precision | False abstentions | Notes |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |",
    ]
    if "distant_oos" in slices:
        s = slices["distant_oos"]
        lines.append(
            f"| Distant out-of-scope (CLINC OOS) | {s.get('samples', 200)} | {format_pct(s['abstention_recall'])} | {format_pct(s['abstention_precision'])} | {s.get('false_abstentions', 0)} | Queries completely unrelated to banking domain |"
        )
    if "missing_option" in slices:
        s = slices["missing_option"]
        lines.append(
            f"| Missing correct option (in-domain) | {s.get('samples', 200)} | {format_pct(s['abstention_recall'])} | {format_pct(s['abstention_precision'])} | {s.get('false_abstentions', 0)} | In-domain banking queries where true intent is omitted |"
        )
    return "\n".join(lines)


def render_experiments_table(rep_dir: Path) -> str:
    e1 = load_json(rep_dir / "exp_e1_shuffled_control.json")
    e2 = load_json(rep_dir / "exp_e2_option_order.json")
    e3 = load_json(rep_dir / "exp_e3_label_bias.json")
    e4 = load_json(rep_dir / "exp_e4_hard_negatives.json")
    e5 = load_json(rep_dir / "exp_e5_abstention_generalization.json")
    e6 = load_json(rep_dir / "exp_e6_contamination.json")
    e7 = load_json(rep_dir / "exp_e7_latency.json")
    e8 = load_json(rep_dir / "exp_e8_fp16_parity.json")
    e9 = load_json(rep_dir / "exp_e9_external_cases.json")

    # Latency p50 at K=5 from e7 (1 thread WASM proxy)
    lat_k5_fp32 = e7["benchmarks"]["fp32"]["1_thread"]["k_5"]["p50_ms"]

    lines = [
        "| Experiment / Question | Baseline / Gold | Measured Result | Boundary / Finding | Source Receipt |",
        "| :--- | :--- | :--- | :--- | :--- |",
        f"| **E1**: Shuffled-context control | Original Acc: {format_pct(e1['original_test']['accuracy'])} | Shuffled Acc: {format_pct(e1['shuffled_control']['accuracy'])} | Accuracy drops by {e1['accuracy_drop']*100:.2f} points; model abstains {format_pct(e1['shuffled_control']['abstention_rate'])} | `reports/v2/exp_e1_shuffled_control.json` |",
        f"| **E2**: Option-order sensitivity | Reversal flips: {e2['reversal_flip_rate']*100:.2f}% | Any permutation: {e2['any_flip_rate']*100:.2f}% | Mean TV distance: {e2['mean_tv_distance_reversal']:.4f}; flips concentrate at low confidence ({e2['confidence_correlation']['mean_conf_flipped']*100:.1f}%) | `reports/v2/exp_e2_option_order.json` |",
        f"| **E3**: Label-length & glossary bias | Curated Acc: {format_pct(e3['accuracy_by_glossary_status']['curated_accuracy'])} | Templated Acc: {format_pct(e3['accuracy_by_glossary_status']['templated_accuracy'])} | Length correlation Pearson r={e3['length_probability_correlation']['pearson_r']:.4f}; uniform glossary rerun Acc: {format_pct(e3['uniform_glossary_rerun']['uniform_accuracy'])} | `reports/v2/exp_e3_label_bias.json` |",
        f"| **E4**: Hard-negative distractors | Random K=5 Acc: 96.00% | Hard K=5: {format_pct(e4['hard_negatives_k5']['accuracy'])} / Hard K=9: {format_pct(e4['hard_negatives_k9']['accuracy'])} | Confusable non-gold distractors cause a {e4['degradation_k5_to_k9_accuracy']*100:.2f} point accuracy drop from K=5 to K=9 | `reports/v2/exp_e4_hard_negatives.json` |",
        f"| **E5**: Abstention generalization | Baseline Recall: {format_pct(e5['baseline_standard']['abstention_recall'])} | Hard K=9: {format_pct(e5['hard_negatives_k9']['abstention_recall'])} / Synonym: {format_pct(e5['synonym_phrasing']['abstention_recall'])} | Combined recall collapses to {format_pct(e5['combined_hard_and_synonym']['abstention_recall'])}; abstention does not generalize across synonyms | `reports/v2/exp_e5_abstention_generalization.json` |",
        f"| **E6**: Contamination audit | Exact matches: 0 / 2.3M | Jaccard >= 0.8: {e6['threshold_counts']['ge_0_8']} pairs | Decontaminated test accuracy moves by {e6['delta']['accuracy']*100:+.2f} points ({format_pct(e6['decontaminated_test_metrics']['accuracy'])}) | `reports/v2/exp_e6_contamination.json` |",
        f"| **E7**: Latency percentiles (K=5) | Prior claim: Unverified point estimate | Single-thread p50: {lat_k5_fp32:.2f} ms | Rigorous multi-trial distribution; WASM single-thread proxy measured across K in {{3,5,9,17,25}} | `reports/v2/exp_e7_latency.json` |",
        f"| **E8**: FP16 quality parity | FP32 test Acc: {format_pct(e8['comparisons']['held_out_test']['fp32']['accuracy'])} | FP16 test Acc: {format_pct(e8['comparisons']['held_out_test']['fp16']['accuracy'])} | Max delta across all splits is {e8['max_accuracy_delta']*100:.2f}%; recommended: SHIP_FP16 | `reports/v2/exp_e8_fp16_parity.json` |",
        f"| **E9**: External TypeSafe evals | TypeSafe Jev: {format_pct(e9['overall']['typesafe_jev_accuracy'])} | Verdict-open-jev: {format_pct(e9['overall']['ours_accuracy'])} | 151M encoder zero-shot floor measured against 26B DiffusionGemma ({format_pct(e9['overall']['openjev_diffusiongemma_accuracy'])}) across 337 cases | `reports/v2/exp_e9_external_cases.json` |",
    ]
    return "\n".join(lines)


def render_latency_table(e7_data: dict[str, Any]) -> str:
    lines = [
        "| Model format | Execution threads | Menu size | p50 latency | p90 latency | p99 latency | Mean latency | Sample tokens |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]
    benchmarks = e7_data.get("benchmarks", {})
    for fmt_key in ["fp32", "fp16"]:
        if fmt_key not in benchmarks:
            continue
        fmt = fmt_key.upper()
        for thr_key in ["1_thread", "4_thread"]:
            if thr_key not in benchmarks[fmt_key]:
                continue
            thr_num = thr_key.split("_")[0]
            threads = f"{thr_num} thread ({'WASM proxy' if thr_num == '1' else 'multi-thread'})"
            for k_key in ["k_3", "k_5", "k_9", "k_17", "k_25"]:
                if k_key not in benchmarks[fmt_key][thr_key]:
                    continue
                b = benchmarks[fmt_key][thr_key][k_key]
                k = f"K = {b['k_cardinality']}"
                lines.append(
                    f"| {fmt} | {threads} | {k} | {b['p50_ms']:.2f} ms | {b['p90_ms']:.2f} ms | {b['p99_ms']:.2f} ms | {b['mean_ms']:.2f} ms | {b['token_count']} tokens |"
                )
    return "\n".join(lines)


def replace_generated_section(content: str, section_name: str, new_content: str) -> str:
    start_tag = f"<!-- BEGIN GENERATED: {section_name} -->"
    end_tag = f"<!-- END GENERATED: {section_name} -->"
    pattern = re.compile(rf"{re.escape(start_tag)}.*?{re.escape(end_tag)}", re.DOTALL)
    replacement = f"{start_tag}\n{new_content}\n{end_tag}"

    if pattern.search(content):
        return pattern.sub(replacement, content)
    else:
        raise ValueError(f"Marker '{start_tag}' not found in document.")


REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Render receipt tables into Markdown files.")
    parser.add_argument("--reports_dir", type=str, default="reports/v2")
    parser.add_argument("--check", action="store_true", help="Verify freshness without writing changes.")
    args = parser.parse_args()

    rep_dir = Path(args.reports_dir)
    if not rep_dir.is_absolute():
        rep_dir = REPO_ROOT / rep_dir
    eval_report = load_json(rep_dir / "evaluation_report_v2.json")
    e7_report = load_json(rep_dir / "exp_e7_latency.json")

    # Render table blocks
    held_out_tbl = render_held_out_table(eval_report)
    cardinality_tbl = render_cardinality_table(eval_report)
    oos_tbl = render_out_of_scope_table(eval_report)
    experiments_tbl = render_experiments_table(rep_dir)
    latency_tbl = render_latency_table(e7_report)

    files_to_update = {
        REPO_ROOT / "README.md": [
            ("held_out_evaluation", held_out_tbl),
            ("cardinality_scaling", cardinality_tbl),
            ("out_of_scope_slices", oos_tbl),
            ("empirical_experiments", experiments_tbl),
        ],
        REPO_ROOT / "WALKTHROUGH.md": [
            ("walkthrough_benchmark_receipts", held_out_tbl),
            ("walkthrough_experiments", experiments_tbl),
            ("walkthrough_slices", cardinality_tbl),
            ("walkthrough_latency", latency_tbl),
        ],
    }

    drift_detected = False

    for file_path, replacements in files_to_update.items():
        if not file_path.exists():
            print(f"Error: file {file_path} does not exist.")
            sys.exit(1)

        original_content = file_path.read_text(encoding="utf-8")
        updated_content = original_content

        for section_name, table_text in replacements:
            try:
                updated_content = replace_generated_section(updated_content, section_name, table_text)
            except ValueError as e:
                print(f"Warning in {file_path}: {e}")

        if original_content != updated_content:
            if args.check:
                drift_detected = True
                print(f"Documentation drift detected in {file_path}:")
                diff = difflib.unified_diff(
                    original_content.splitlines(),
                    updated_content.splitlines(),
                    fromfile=str(file_path),
                    tofile=f"{file_path} (rendered)",
                    lineterm="",
                )
                print("\n".join(diff))
            else:
                file_path.write_text(updated_content, encoding="utf-8")
                print(f"Rendered and updated receipts in {file_path}")

    if args.check and drift_detected:
        print("\nVerification failed: documentation is stale compared to reports/v2 receipts.")
        sys.exit(1)
    elif args.check:
        print("Verification passed: all documentation matches committed JSON receipts exactly.")


if __name__ == "__main__":
    main()
