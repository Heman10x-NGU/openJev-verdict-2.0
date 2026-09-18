"""Evaluate OpenJev-ModernBERT calibration, accuracy, abstention, and slices.

Fits positive scalar temperature scaling via L-BFGS on held-out calibration logits,
evaluates held-out test data and challenge slices under strictly proper scoring rules
(Brier score, NLL), and calculates Expected Calibration Error (ECE), selective risk,
and coverage curves.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import AutoTokenizer

from core.calibration import (
    TemperatureCalibrator,
    brier_score_loss,
    compute_bootstrap_ci,
    compute_ece,
    cross_entropy_loss,
)
from core.formatting import build_model_input, format_prompt
from gliclass import GLiClassModel


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def extract_logits_and_targets(
    model: GLiClassModel,
    tokenizer: Any,
    records: list[dict[str, Any]],
    device: torch.device,
    batch_size: int = 16,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    """Run forward passes and collect candidate logits and target indices."""
    model.eval()
    all_logits = []
    all_targets = []
    metadata = []

    with torch.no_grad():
        for i in range(0, len(records), batch_size):
            batch = records[i : i + batch_size]
            batch_labels = [
                [c["description"] for c in r["candidates"]] for r in batch
            ]
            batch_ids = [[c["id"] for c in r["candidates"]] for r in batch]
            target_ids = [r["target_id"] for r in batch]

            prompts = [
                build_model_input(r["question"], r["text"], labels)
                for r, labels in zip(batch, batch_labels)
            ]
            tokens = tokenizer(
                prompts, padding=True, truncation=True, return_tensors="pt"
            ).to(device)

            outputs = model(**tokens)
            raw_logits = outputs.logits

            for b_idx in range(len(batch)):
                labels = batch_labels[b_idx]
                k_classes = len(labels)
                c_logits = raw_logits[b_idx, :k_classes].float().cpu()
                c_ids = batch_ids[b_idx]
                target_id = target_ids[b_idx]
                target_idx = c_ids.index(target_id)

                all_logits.append(c_logits)
                all_targets.append(target_idx)
                metadata.append(
                    {
                        "id": batch[b_idx]["id"],
                        "text": batch[b_idx]["text"],
                        "target_id": target_id,
                        "candidate_ids": c_ids,
                        "is_abstention": batch[b_idx].get("is_abstention", False),
                        "abstention_subtype": batch[b_idx].get("abstention_subtype", "unknown"),
                    }
                )

    # Pad logits for batched tensor math
    max_k = max(l.shape[0] for l in all_logits)
    padded_logits = torch.zeros((len(all_logits), max_k), dtype=torch.float32)
    targets_tensor = torch.tensor(all_targets, dtype=torch.long)

    for idx, l in enumerate(all_logits):
        padded_logits[idx, : l.shape[0]] = l
        if l.shape[0] < max_k:
            padded_logits[idx, l.shape[0] :] = float("-inf")

    return padded_logits, targets_tensor, metadata


def compute_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    metadata: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute accuracy, proper scoring losses, ECE, and selective risk."""
    num_samples = len(targets)
    probs = F.softmax(logits, dim=-1).cpu().numpy()
    preds = np.argmax(probs, axis=-1)
    confs = np.max(probs, axis=-1)
    targets_np = targets.cpu().numpy()

    corrects = (preds == targets_np).astype(np.float32)
    top1_acc = float(np.mean(corrects))

    # Mask valid options for loss calculation
    valid_mask = torch.isfinite(logits)
    nll = float(cross_entropy_loss(logits, targets, valid_mask=valid_mask).item())
    brier = float(brier_score_loss(logits, targets, valid_mask=valid_mask).item())

    ece_equal_width = compute_ece(
        confs, corrects, n_bins=10, strategy="equal_width", min_bin_count=10
    )
    ece_equal_mass = compute_ece(
        confs, corrects, n_bins=10, strategy="equal_mass", min_bin_count=10
    )
    ece_adaptive = compute_ece(
        confs, corrects, n_bins=10, strategy="adaptive", min_bin_count=10
    )

    # Abstention metrics
    abstention_tp = 0
    abstention_fp = 0
    abstention_fn = 0
    is_abstain_preds = []
    is_abstain_golds = []
    for idx, meta in enumerate(metadata):
        pred_is_abstain = meta["candidate_ids"][preds[idx]] == "__insufficient_evidence__"
        actual_is_abstain = meta["is_abstention"]
        is_abstain_preds.append(pred_is_abstain)
        is_abstain_golds.append(actual_is_abstain)

        if pred_is_abstain and actual_is_abstain:
            abstention_tp += 1
        elif pred_is_abstain and not actual_is_abstain:
            abstention_fp += 1
        elif not pred_is_abstain and actual_is_abstain:
            abstention_fn += 1

    is_abstain_preds_arr = np.array(is_abstain_preds, dtype=bool)
    is_abstain_golds_arr = np.array(is_abstain_golds, dtype=bool)

    precision = (
        abstention_tp / (abstention_tp + abstention_fp)
        if (abstention_tp + abstention_fp) > 0
        else 0.0
    )
    recall = (
        abstention_tp / (abstention_tp + abstention_fn)
        if (abstention_tp + abstention_fn) > 0
        else 0.0
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    # 1000-resample bootstrap 95% CIs
    indices = np.arange(num_samples)
    ci_acc = compute_bootstrap_ci(
        indices, lambda idxs: float(np.mean(corrects[idxs])), n_resamples=1000
    )
    ci_ece_ew = compute_bootstrap_ci(
        indices,
        lambda idxs: compute_ece(
            confs[idxs], corrects[idxs], n_bins=10, strategy="equal_width", min_bin_count=10
        ).ece,
        n_resamples=1000,
    )
    ci_ece_em = compute_bootstrap_ci(
        indices,
        lambda idxs: compute_ece(
            confs[idxs], corrects[idxs], n_bins=10, strategy="equal_mass", min_bin_count=10
        ).ece,
        n_resamples=1000,
    )

    def abs_recall_fn(idxs: np.ndarray) -> float:
        tp = np.sum(is_abstain_preds_arr[idxs] & is_abstain_golds_arr[idxs])
        pos = np.sum(is_abstain_golds_arr[idxs])
        return float(tp / pos) if pos > 0 else 0.0

    def abs_prec_fn(idxs: np.ndarray) -> float:
        tp = np.sum(is_abstain_preds_arr[idxs] & is_abstain_golds_arr[idxs])
        pred_pos = np.sum(is_abstain_preds_arr[idxs])
        return float(tp / pred_pos) if pred_pos > 0 else 0.0

    ci_abs_recall = compute_bootstrap_ci(indices, abs_recall_fn, n_resamples=1000)
    ci_abs_precision = compute_bootstrap_ci(indices, abs_prec_fn, n_resamples=1000)

    # Coverage vs Selective Risk across thresholds
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.85, 0.90, 0.95, 0.98]
    selective_curves = []
    for tau in thresholds:
        accepted_mask = confs >= tau
        accepted_count = int(np.sum(accepted_mask))
        coverage = accepted_count / num_samples if num_samples > 0 else 0.0
        if accepted_count > 0:
            errors_in_accepted = int(np.sum(corrects[accepted_mask] == 0))
            risk = errors_in_accepted / accepted_count
            # Wilson / Clopper-Pearson 95% upper bound approximation
            z = 1.96
            denom = 1 + (z**2) / accepted_count
            centre = (risk + (z**2) / (2 * accepted_count)) / denom
            spread = (
                z
                * math.sqrt((risk * (1 - risk) + (z**2) / (4 * accepted_count)) / accepted_count)
                / denom
            )
            risk_ub_95 = min(1.0, centre + spread)
        else:
            errors_in_accepted = 0
            risk = 0.0
            risk_ub_95 = 0.0

        selective_curves.append(
            {
                "threshold": tau,
                "coverage": float(coverage),
                "accepted_count": accepted_count,
                "errors": errors_in_accepted,
                "selective_risk": float(risk),
                "risk_upper_bound_95": float(risk_ub_95),
            }
        )

    return {
        "accuracy": top1_acc,
        "negative_log_likelihood": nll,
        "brier_score": brier,
        "ece_equal_width": ece_equal_width.ece,
        "mce_equal_width": ece_equal_width.mce,
        "ece_equal_mass": ece_equal_mass.ece,
        "mce_equal_mass": ece_equal_mass.mce,
        "ece_adaptive": ece_adaptive.ece,
        "mce_adaptive": ece_adaptive.mce,
        "ci_95": {
            "accuracy": list(ci_acc),
            "ece_equal_width": list(ci_ece_ew),
            "ece_equal_mass": list(ci_ece_em),
        },
        "abstention": {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "false_abstentions": abstention_fp,
            "true_abstentions": abstention_tp,
            "false_negatives": abstention_fn,
            "total_gold_abstentions": abstention_tp + abstention_fn,
            "total_in_scope": num_samples - (abstention_tp + abstention_fn),
            "ci_95": {
                "recall": list(ci_abs_recall),
                "precision": list(ci_abs_precision),
            },
        },
        "selective_policy": selective_curves,
        "reliability_bins": {
            "edges": list(ece_equal_width.bin_edges),
            "accuracies": [
                float(acc) if count > 0 else None
                for acc, count in zip(ece_equal_width.bin_accuracies, ece_equal_width.bin_counts)
            ],
            "confidences": [
                float(conf) if count > 0 else None
                for conf, count in zip(ece_equal_width.bin_confidences, ece_equal_width.bin_counts)
            ],
            "counts": list(ece_equal_width.bin_counts),
            "min_bin_count": ece_equal_width.min_bin_count,
        },
    }


def evaluate_pipeline(
    cal_file: str = "data/real_banking_cal.jsonl",
    test_file: str = "data/real_banking_test.jsonl",
    checkpoint_dir: str = "artifacts",
    reports_dir: str = "reports",
    model_name: str = "knowledgator/gliclass-modern-base-v2.0",
    device_name: str = "mps",
) -> dict[str, Any]:
    """Fit temperature on cal set, evaluate test set and slices, and output predictions."""
    rep_path = Path(reports_dir)
    rep_path.mkdir(parents=True, exist_ok=True)

    if device_name == "mps" and not torch.backends.mps.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(device_name)

    print(f"Loading model on {device.type.upper()}...")
    model = GLiClassModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    checkpoint_candidates = [
        Path(checkpoint_dir) / "model.safetensors",
        Path(checkpoint_dir) / "openjev_modernbert.safetensors",
    ]
    checkpoint_path = None
    for cp in checkpoint_candidates:
        if cp.exists():
            checkpoint_path = cp
            break
    if checkpoint_path is None:
        raise FileNotFoundError(f"Required checkpoint not found in {checkpoint_dir}")

    param_count = sum(p.numel() for p in model.parameters())
    param_str = f"{param_count / 1e6:.1f}M"
    print(f"Loading fine-tuned weights from {checkpoint_path} ({param_count:,} parameters, ~{param_str})...")
    state_dict = load_file(str(checkpoint_path))
    model.load_state_dict(state_dict)
    model.to(device)

    # 1. Temperature scaling on calibration split
    print(f"Extracting calibration logits from {cal_file}...")
    cal_records = load_jsonl(Path(cal_file))
    cal_logits, cal_targets, _ = extract_logits_and_targets(
        model, tokenizer, cal_records, device
    )

    print("Fitting Temperature Calibrator via L-BFGS on calibration set...")
    valid_cal_mask = torch.isfinite(cal_logits)
    calibrator = TemperatureCalibrator(
        model_id=f"verdict-open-jev-modernbert-{param_str.lower()}",
        scope="restricted_5_candidate_selection",
        artifact_hash="e6a13d7ebffc39050d5e8687a4192b02fc922262d4e8bb43586ab32605ad2ea8",
    )
    calibrator.fit(cal_logits, cal_targets, valid_mask=valid_cal_mask)
    optimal_temp = calibrator.temperature
    print(f"Optimal Temperature: T = {optimal_temp:.4f}")
    calibrator.save(Path(checkpoint_dir) / "calibrator_modernbert.json")
    calibrator.save(Path(checkpoint_dir) / "calibrator.json")

    # 2. Main Held-Out Test evaluation
    print(f"Extracting held-out test logits from {test_file}...")
    test_records = load_jsonl(Path(test_file))
    test_logits, test_targets, test_meta = extract_logits_and_targets(
        model, tokenizer, test_records, device
    )

    uncalibrated_metrics = compute_metrics(test_logits, test_targets, test_meta)
    with torch.no_grad():
        calibrated_test_logits = calibrator(test_logits)
    calibrated_metrics = compute_metrics(
        calibrated_test_logits, test_targets, test_meta
    )

    # 3. Challenge Slices
    slices_results = {}
    slice_files = {
        "missing_option": "data/slice_missing_option.jsonl",
        "distant_oos": "data/slice_distant_oos.jsonl",
        "cardinality_k3": "data/slice_cardinality_k3.jsonl",
        "cardinality_k5": "data/slice_cardinality_k5.jsonl",
        "cardinality_k9": "data/slice_cardinality_k9.jsonl",
        "cardinality_k17": "data/slice_cardinality_k17.jsonl",
        "cardinality_k25": "data/slice_cardinality_k25.jsonl",
    }

    failure_examples = []

    for s_name, s_file in slice_files.items():
        p = Path(s_file)
        if not p.exists():
            continue
        print(f"Evaluating challenge slice: {s_name} ({s_file})...")
        s_records = load_jsonl(p)
        s_logits, s_targets, s_meta = extract_logits_and_targets(
            model, tokenizer, s_records, device
        )
        with torch.no_grad():
            s_cal_logits = calibrator(s_logits)
        s_metrics = compute_metrics(s_cal_logits, s_targets, s_meta)
        slices_results[s_name] = {
            "samples": len(s_records),
            "accuracy": s_metrics["accuracy"],
            "brier": s_metrics["brier_score"],
            "ece": s_metrics["ece_equal_width"],
            "abstention_recall": s_metrics["abstention"]["recall"],
            "abstention_precision": s_metrics["abstention"]["precision"],
            "abstention_f1": s_metrics["abstention"]["f1_score"],
            "false_abstentions": s_metrics["abstention"]["false_abstentions"],
        }

        # Inspect failure cases for failure gallery
        s_probs = F.softmax(s_cal_logits, dim=-1).cpu().numpy()
        s_preds = np.argmax(s_probs, axis=-1)
        for idx, meta in enumerate(s_meta):
            if s_preds[idx] != s_targets[idx].item() and len(failure_examples) < 20:
                pred_cand = meta["candidate_ids"][s_preds[idx]]
                pred_prob = float(s_probs[idx, s_preds[idx]])
                failure_examples.append(
                    {
                        "slice": s_name,
                        "id": meta["id"],
                        "text": meta["text"],
                        "expected": meta["target_id"],
                        "predicted": pred_cand,
                        "confidence": pred_prob,
                    }
                )

    # 4. Save per-prediction JSONL for verification
    pred_path = rep_path / "predictions_v2.jsonl"
    test_probs = F.softmax(calibrated_test_logits, dim=-1).cpu().numpy()
    test_preds = np.argmax(test_probs, axis=-1)

    total_test_errors = 0
    errors_above_85 = 0
    errors_below_85 = 0

    with open(pred_path, "w", encoding="utf-8") as f:
        for idx, meta in enumerate(test_meta):
            is_correct = bool(test_preds[idx] == test_targets[idx].item())
            conf = float(test_probs[idx, test_preds[idx]])
            pred_id = meta["candidate_ids"][test_preds[idx]]
            row = {
                "id": meta["id"],
                "target_id": meta["target_id"],
                "predicted_id": pred_id,
                "confidence": conf,
                "is_correct": is_correct,
                "is_abstention": meta["is_abstention"],
                "above_85_threshold": bool(conf >= 0.85),
            }
            f.write(json.dumps(row) + "\n")
            if not is_correct:
                total_test_errors += 1
                if conf >= 0.85:
                    errors_above_85 += 1
                else:
                    errors_below_85 += 1

    print(f"Saved prediction records to {pred_path}")

    # 5. Generate failure_gallery.md
    fg_path = rep_path / "failure_gallery.md"
    pct_above = (errors_above_85 / total_test_errors * 100.0) if total_test_errors > 0 else 0.0
    pct_below = (errors_below_85 / total_test_errors * 100.0) if total_test_errors > 0 else 0.0

    # Collect test set failures
    test_errors = []
    for idx, meta in enumerate(test_meta):
        if test_preds[idx] != test_targets[idx].item():
            p_conf = float(test_probs[idx, test_preds[idx]])
            test_errors.append(
                {
                    "slice": "held_out_test",
                    "id": meta["id"],
                    "text": meta["text"],
                    "expected": meta["target_id"],
                    "predicted": meta["candidate_ids"][test_preds[idx]],
                    "confidence": p_conf,
                    "above_85": p_conf >= 0.85,
                }
            )

    test_errors.sort(key=lambda x: x["confidence"], reverse=True)

    gallery_cases = []
    # Sample top high-confidence failures (>= 0.85)
    gallery_cases.extend([c for c in test_errors if c["confidence"] >= 0.85][:8])
    # Sample medium-confidence failures (0.50 - 0.85)
    gallery_cases.extend([c for c in test_errors if 0.50 <= c["confidence"] < 0.85][:6])
    # Sample low-confidence failures (< 0.50)
    gallery_cases.extend([c for c in test_errors if c["confidence"] < 0.50][:4])
    # Add challenge slice failures
    for fe in failure_examples[:8]:
        gallery_cases.append(
            {
                "slice": fe["slice"],
                "id": fe["id"],
                "text": fe["text"],
                "expected": fe["expected"],
                "predicted": fe["predicted"],
                "confidence": fe["confidence"],
                "above_85": fe["confidence"] >= 0.85,
            }
        )

    with open(fg_path, "w", encoding="utf-8") as f:
        f.write("# Decision Engine Failure Gallery and Boundary Audit\n\n")
        f.write(
            "Documenting empirical model errors, false acceptances, and boundary edge cases "
            "across held-out test data and challenge slices.\n\n"
        )
        f.write("| Slice / Split | Case ID | Customer Utterance | Expected | Predicted | Confidence | Above 85% Gate |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for fe in gallery_cases:
            above_str = "Yes (Bypasses Gate)" if fe["above_85"] else "No (Safely Gated)"
            clean_text = fe["text"].replace("\n", " ")[:48]
            f.write(
                f"| `{fe['slice']}` | `{fe['id']}` | {clean_text}... | "
                f"`{fe['expected']}` | `{fe['predicted']}` | {fe['confidence']*100:.1f}% | {above_str} |\n"
            )
        f.write("\n### Empirical Gating Analysis\n\n")
        f.write(
            f"Of the {total_test_errors} total test set classification errors "
            f"({total_test_errors}/{len(test_meta)} = {total_test_errors/len(test_meta)*100:.2f}% error rate):\n\n"
        )
        f.write(
            f"1. Errors above 85% confidence: {errors_above_85} ({pct_above:.1f}% of errors) "
            "exhibited confidence >= 85% and would bypass an autonomous policy threshold of 85%.\n"
        )
        f.write(
            f"2. Errors below 85% confidence: {errors_below_85} ({pct_below:.1f}% of errors) "
            "were produced with confidence < 85% and would be safely routed to human supervisor review.\n\n"
        )
        f.write(
            "An autonomous threshold at 85% intercepts the majority of lower-confidence errors, "
            "but near-sibling confusions frequently exhibit overconfident incorrect predictions.\n"
        )
    print(f"Saved failure gallery to {fg_path}")

    # Save comprehensive report
    report = {
        "model": "Verdict-open-jev-ModernBERT",
        "parameters": param_count,
        "parameters_display": param_str,
        "checkpoint": str(checkpoint_path),
        "temperature": optimal_temp,
        "uncalibrated": uncalibrated_metrics,
        "calibrated": calibrated_metrics,
        "slices": slices_results,
        "gate_85_audit": {
            "total_test_errors": total_test_errors,
            "errors_above_85": errors_above_85,
            "errors_below_85": errors_below_85,
            "pct_above_85": pct_above,
            "pct_below_85": pct_below,
        },
    }

    report_file = rep_path / "evaluation_report_v2.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Saved full evaluation report to {report_file}")

    print("\n================ BENCHMARK EVALUATION (UNCALIBRATED vs CALIBRATED) ================")
    print(f"{'Metric':<30} | {'Uncalibrated':<15} | {'Calibrated':<15} | {'95% Bootstrap CI':<20}")
    print("-" * 88)
    ci_acc = calibrated_metrics['ci_95']['accuracy']
    print(f"{'Top-1 Accuracy':<30} | {uncalibrated_metrics['accuracy']*100:6.2f}%         | {calibrated_metrics['accuracy']*100:6.2f}%         | [{ci_acc[0]*100:.2f}%, {ci_acc[1]*100:.2f}%]")
    print(f"{'Negative Log-Likelihood':<30} | {uncalibrated_metrics['negative_log_likelihood']:6.4f}          | {calibrated_metrics['negative_log_likelihood']:6.4f}          | N/A")
    print(f"{'Brier Score':<30} | {uncalibrated_metrics['brier_score']:6.4f}          | {calibrated_metrics['brier_score']:6.4f}          | N/A")
    ci_ece = calibrated_metrics['ci_95']['ece_equal_width']
    print(f"{'ECE (Equal-Width, 10 bins)':<30} | {uncalibrated_metrics['ece_equal_width']*100:6.2f}%         | {calibrated_metrics['ece_equal_width']*100:6.2f}%         | [{ci_ece[0]*100:.2f}%, {ci_ece[1]*100:.2f}%]")
    ci_eem = calibrated_metrics['ci_95']['ece_equal_mass']
    print(f"{'ECE (Equal-Mass, 10 bins)':<30} | {uncalibrated_metrics['ece_equal_mass']*100:6.2f}%         | {calibrated_metrics['ece_equal_mass']*100:6.2f}%         | [{ci_eem[0]*100:.2f}%, {ci_eem[1]*100:.2f}%]")
    print(f"{'ECE (Adaptive, min=10)':<30} | {uncalibrated_metrics['ece_adaptive']*100:6.2f}%         | {calibrated_metrics['ece_adaptive']*100:6.2f}%         | N/A")
    print(f"{'MCE (Equal-Width, min=10)':<30} | {uncalibrated_metrics['mce_equal_width']*100:6.2f}%         | {calibrated_metrics['mce_equal_width']*100:6.2f}%         | N/A")
    ci_rec = calibrated_metrics['abstention']['ci_95']['recall']
    print(f"{'Abstention Recall':<30} | {uncalibrated_metrics['abstention']['recall']*100:6.2f}%         | {calibrated_metrics['abstention']['recall']*100:6.2f}%         | [{ci_rec[0]*100:.2f}%, {ci_rec[1]*100:.2f}%]")
    ci_pre = calibrated_metrics['abstention']['ci_95']['precision']
    print(f"{'Abstention Precision':<30} | {uncalibrated_metrics['abstention']['precision']*100:6.2f}%         | {calibrated_metrics['abstention']['precision']*100:6.2f}%         | [{ci_pre[0]*100:.2f}%, {ci_pre[1]*100:.2f}%]")
    print(f"{'Abstention F1':<30} | {uncalibrated_metrics['abstention']['f1_score']*100:6.2f}%         | {calibrated_metrics['abstention']['f1_score']*100:6.2f}%         | N/A")
    print(f"{'False Abstentions (in-scope)':<30} | {uncalibrated_metrics['abstention']['false_abstentions']:<15} | {calibrated_metrics['abstention']['false_abstentions']:<15} | N/A")
    print("-------------------------------- Slices Summary --------------------------------")
    for s_name, res in slices_results.items():
        print(f"Slice [{s_name:18s}]: Acc={res['accuracy']*100:5.1f}% | Recall={res['abstention_recall']*100:5.1f}% | Precision={res['abstention_precision']*100:5.1f}% | FalseAbs={res['false_abstentions']:<3} | ECE={res['ece']:.4f}")
    print("================================================================================")

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Verdict-open-jev-ModernBERT.")
    parser.add_argument("--cal_file", type=str, default="data/real_banking_cal.jsonl")
    parser.add_argument("--test_file", type=str, default="data/real_banking_test.jsonl")
    parser.add_argument("--checkpoint_dir", type=str, default="artifacts")
    parser.add_argument("--reports_dir", type=str, default="reports")
    parser.add_argument("--model_name", type=str, default="knowledgator/gliclass-modern-base-v2.0")
    parser.add_argument("--device", type=str, default="mps")
    args = parser.parse_args()

    evaluate_pipeline(
        cal_file=args.cal_file,
        test_file=args.test_file,
        checkpoint_dir=args.checkpoint_dir,
        reports_dir=args.reports_dir,
        model_name=args.model_name,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
