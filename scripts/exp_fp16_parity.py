"""Experiment E8: FP16 Quality and Parity Benchmark.

Evaluates FP32 versus FP16 ONNX models across the full held-out test set
and all challenge slices (missing option, distant OOS, cardinalities K=3, 5, 9, 17, 25).
Computes accuracy, NLL, Brier score, ECE, and delta between FP32 and FP16.
Verifies the acceptance criterion: accuracy shift must be <= 0.5 percentage points.

Writes receipt to reports/v2/exp_e8_fp16_parity.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from transformers import AutoTokenizer

from core.calibration import brier_score_loss, compute_ece, cross_entropy_loss
from core.formatting import build_model_input

try:
    import onnxruntime as ort
    import torch
    import torch.nn.functional as F
    ORT_AVAILABLE = True
except ImportError:
    ORT_AVAILABLE = False


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def evaluate_onnx_model(
    session: ort.InferenceSession,
    tokenizer: Any,
    records: list[dict[str, Any]],
    batch_size: int = 16,
) -> dict[str, float]:
    all_logits = []
    all_targets = []
    correct = 0
    abs_correct = 0
    abs_total = 0

    question = "Classify the following query into the appropriate category:"

    for i in range(0, len(records), batch_size):
        batch = records[i : i + batch_size]
        batch_labels = [[c["description"] for c in r["candidates"]] for r in batch]
        batch_ids = [[c["id"] for c in r["candidates"]] for r in batch]
        target_ids = [r["target_id"] for r in batch]

        prompts = [
            build_model_input(question, r["text"], labels)
            for r, labels in zip(batch, batch_labels)
        ]
        tokens = tokenizer(prompts, padding=True, truncation=True, return_tensors="np")
        ort_inputs = {
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"],
        }
        ort_outs = session.run(None, ort_inputs)
        raw_logits = ort_outs[0]

        for b_idx in range(len(batch)):
            k = len(batch_labels[b_idx])
            c_logits = raw_logits[b_idx, :k]
            pred_idx = int(np.argmax(c_logits))
            target_idx = batch_ids[b_idx].index(target_ids[b_idx])

            if pred_idx == target_idx:
                correct += 1

            if batch[b_idx].get("is_abstention", False):
                abs_total += 1
                if pred_idx == target_idx:
                    abs_correct += 1

            all_logits.append(torch.tensor(c_logits, dtype=torch.float32))
            all_targets.append(target_idx)

    max_k = max(l.shape[-1] for l in all_logits)
    padded = torch.zeros((len(all_logits), max_k), dtype=torch.float32)
    for idx, l in enumerate(all_logits):
        padded[idx, : l.shape[-1]] = l
        if l.shape[-1] < max_k:
            padded[idx, l.shape[-1] :] = float("-inf")

    targets_tensor = torch.tensor(all_targets, dtype=torch.long)
    valid_mask = torch.isfinite(padded)
    nll = float(cross_entropy_loss(padded, targets_tensor, valid_mask=valid_mask).item())
    brier = float(brier_score_loss(padded, targets_tensor, valid_mask=valid_mask).item())

    probs_all = F.softmax(padded, dim=-1).numpy()
    preds_all = np.argmax(probs_all, axis=-1)
    confs_all = np.max(probs_all, axis=-1)
    corrects_all = (preds_all == targets_tensor.numpy()).astype(np.float32)
    ece = compute_ece(confs_all, corrects_all, n_bins=10, strategy="equal_width").ece

    return {
        "samples": len(records),
        "accuracy": correct / len(records),
        "nll": nll,
        "brier": brier,
        "ece": ece,
        "abstention_recall": (abs_correct / abs_total) if abs_total > 0 else 0.0,
    }


def run_fp16_parity_experiment(
    checkpoint_dir: str = "artifacts/v2",
    reports_dir: str = "reports/v2",
    model_name: str = "knowledgator/gliclass-modern-base-v2.0",
) -> dict[str, Any]:
    if not ORT_AVAILABLE:
        raise RuntimeError("onnxruntime and torch are required for FP16 parity evaluation.")

    rep_path = Path(reports_dir)
    rep_path.mkdir(parents=True, exist_ok=True)

    fp32_path = Path(checkpoint_dir) / "model.onnx"
    if not fp32_path.exists():
        fp32_path = Path(checkpoint_dir) / "openjev_modernbert.onnx"
    fp16_path = Path(checkpoint_dir) / "model_fp16.onnx"

    if not fp32_path.exists() or not fp16_path.exists():
        raise FileNotFoundError(f"Both FP32 ({fp32_path}) and FP16 ({fp16_path}) ONNX models must exist.")

    print(f"Loading FP32 model: {fp32_path}")
    sess_fp32 = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
    print(f"Loading FP16 model: {fp16_path}")
    sess_fp16 = ort.InferenceSession(str(fp16_path), providers=["CPUExecutionProvider"])

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    eval_datasets = {
        "held_out_test": "data/real_banking_test.jsonl",
        "missing_option": "data/slice_missing_option.jsonl",
        "distant_oos": "data/slice_distant_oos.jsonl",
        "cardinality_k3": "data/slice_cardinality_k3.jsonl",
        "cardinality_k5": "data/slice_cardinality_k5.jsonl",
        "cardinality_k9": "data/slice_cardinality_k9.jsonl",
        "cardinality_k17": "data/slice_cardinality_k17.jsonl",
        "cardinality_k25": "data/slice_cardinality_k25.jsonl",
    }

    comparison_results = {}
    max_acc_delta = 0.0

    for d_name, d_file in eval_datasets.items():
        p = Path(d_file)
        if not p.exists():
            continue
        print(f"Evaluating {d_name} ({d_file})...")
        records = load_jsonl(p)
        res_fp32 = evaluate_onnx_model(sess_fp32, tokenizer, records)
        res_fp16 = evaluate_onnx_model(sess_fp16, tokenizer, records)

        acc_delta = res_fp16["accuracy"] - res_fp32["accuracy"]
        max_acc_delta = max(max_acc_delta, abs(acc_delta))

        comparison_results[d_name] = {
            "samples": len(records),
            "fp32": res_fp32,
            "fp16": res_fp16,
            "delta": {
                "accuracy": acc_delta,
                "nll": res_fp16["nll"] - res_fp32["nll"],
                "brier": res_fp16["brier"] - res_fp32["brier"],
                "ece": res_fp16["ece"] - res_fp32["ece"],
            },
        }

        print(
            f"  {d_name:<18s} -> FP32 Acc: {res_fp32['accuracy']*100:5.2f}% | "
            f"FP16 Acc: {res_fp16['accuracy']*100:5.2f}% (delta: {acc_delta*100:+5.2f}%)"
        )

    parity_maintained = bool(max_acc_delta <= 0.005)
    receipt = {
        "experiment": "E8_fp16_parity",
        "fp32_file": fp32_path.name,
        "fp16_file": fp16_path.name,
        "max_accuracy_delta": max_acc_delta,
        "parity_threshold": 0.005,
        "parity_maintained": parity_maintained,
        "recommendation": "ship_fp16" if parity_maintained else "ship_fp32",
        "comparisons": comparison_results,
    }

    out_file = rep_path / "exp_e8_fp16_parity.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(receipt, f, indent=2)

    print(f"\nSaved E8 receipt to {out_file}")
    print(f"Max Accuracy Delta across splits: {max_acc_delta*100:.3f}% (Threshold: <= 0.50%)")
    print(f"Parity Maintained: {parity_maintained} -> Recommendation: {receipt['recommendation'].upper()}")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description="Run E8 FP16 Parity Benchmark.")
    parser.add_argument("--checkpoint_dir", type=str, default="artifacts/v2")
    parser.add_argument("--reports_dir", type=str, default="reports/v2")
    parser.add_argument("--model_name", type=str, default="knowledgator/gliclass-modern-base-v2.0")
    args = parser.parse_args()

    run_fp16_parity_experiment(
        checkpoint_dir=args.checkpoint_dir,
        reports_dir=args.reports_dir,
        model_name=args.model_name,
    )


if __name__ == "__main__":
    main()
