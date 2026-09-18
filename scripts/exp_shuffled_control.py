"""Experiment E1: Shuffled-Context Control Baseline.

Tests whether the decision model genuinely conditions on input context or exploits
label priors. Pairs each held-out test record's candidate option menu with a different
record's context (seeded permutation). Evaluates accuracy, NLL, Brier score, and
abstention rate, writing a verified JSON receipt to reports/v2/.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import AutoTokenizer

from core.calibration import TemperatureCalibrator, brier_score_loss, cross_entropy_loss
from core.formatting import build_model_input
from gliclass import GLiClassModel


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def run_shuffled_control(
    test_file: str = "data/real_banking_test.jsonl",
    checkpoint_dir: str = "artifacts/v2",
    reports_dir: str = "reports/v2",
    model_name: str = "knowledgator/gliclass-modern-base-v2.0",
    device_name: str = "mps",
    seed: int = 42,
    batch_size: int = 16,
) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    rep_path = Path(reports_dir)
    rep_path.mkdir(parents=True, exist_ok=True)

    if device_name == "mps" and not torch.backends.mps.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(device_name)

    print(f"Loading test records from {test_file}...")
    records = load_jsonl(Path(test_file))
    n_records = len(records)
    if n_records == 0:
        raise ValueError("Test set is empty.")

    # Create derangement / seeded permutation of contexts
    shuffled_indices = list(range(n_records))
    rng = random.Random(seed)
    # Ensure no context matches original record (derangement)
    for i in range(n_records - 1, 0, -1):
        j = rng.randint(0, i - 1)
        shuffled_indices[i], shuffled_indices[j] = shuffled_indices[j], shuffled_indices[i]

    shuffled_records = []
    for orig_idx, perm_idx in enumerate(shuffled_indices):
        orig_rec = records[orig_idx]
        shuffled_context = records[perm_idx]["text"]
        shuffled_records.append(
            {
                "id": orig_rec["id"],
                "question": orig_rec["question"],
                "text": shuffled_context,  # Swapped context
                "original_text": orig_rec["text"],
                "candidates": orig_rec["candidates"],
                "target_id": orig_rec["target_id"],
                "is_abstention": orig_rec.get("is_abstention", False),
            }
        )

    # Load model
    print(f"Loading model on {device.type.upper()}...")
    model = GLiClassModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    checkpoint_candidates = [
        Path(checkpoint_dir) / "model.safetensors",
        Path(checkpoint_dir) / "openjev_modernbert.safetensors",
    ]
    ckpt_path = next((cp for cp in checkpoint_candidates if cp.exists()), None)
    if ckpt_path is None:
        raise FileNotFoundError(f"Checkpoint not found in {checkpoint_dir}")

    state_dict = load_file(str(ckpt_path))
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Load calibrator if available
    cal_path = Path(checkpoint_dir) / "calibrator.json"
    if not cal_path.exists():
        cal_path = Path(checkpoint_dir) / "calibrator_modernbert.json"
    calibrator = None
    if cal_path.exists():
        calibrator = TemperatureCalibrator.load(cal_path)
        print(f"Loaded calibrator (T = {calibrator.temperature:.4f})")

    # Evaluate on both original and shuffled records
    def evaluate_set(eval_records: list[dict[str, Any]]) -> dict[str, float]:
        all_logits = []
        all_targets = []
        all_preds = []
        all_probs = []
        abstention_preds = 0
        total_correct = 0

        with torch.no_grad():
            for i in range(0, len(eval_records), batch_size):
                batch = eval_records[i : i + batch_size]
                batch_labels = [[c["description"] for c in r["candidates"]] for r in batch]
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
                    k_classes = len(batch_labels[b_idx])
                    c_logits = raw_logits[b_idx, :k_classes].float().unsqueeze(0)
                    if calibrator is not None:
                        c_logits = calibrator(c_logits)

                    probs = F.softmax(c_logits, dim=-1).squeeze(0).cpu().numpy()
                    pred_idx = int(np.argmax(probs))
                    target_idx = batch_ids[b_idx].index(target_ids[b_idx])

                    if pred_idx == target_idx:
                        total_correct += 1

                    cand_id = batch_ids[b_idx][pred_idx]
                    if cand_id == "__insufficient_evidence__":
                        abstention_preds += 1

                    all_logits.append(c_logits.cpu())
                    all_targets.append(target_idx)
                    all_preds.append(pred_idx)
                    all_probs.append(float(probs[pred_idx]))

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
        acc = total_correct / len(eval_records)
        abs_rate = abstention_preds / len(eval_records)
        mean_conf = float(np.mean(all_probs))

        return {
            "accuracy": acc,
            "negative_log_likelihood": nll,
            "brier_score": brier,
            "mean_confidence": mean_conf,
            "abstention_rate": abs_rate,
        }

    print("Evaluating original (unshuffled) test set...")
    orig_metrics = evaluate_set(records)
    print("Evaluating shuffled-context control set...")
    shuffled_metrics = evaluate_set(shuffled_records)

    receipt = {
        "experiment": "E1_shuffled_context_control",
        "sample_count": n_records,
        "seed": seed,
        "original_test": orig_metrics,
        "shuffled_control": shuffled_metrics,
        "accuracy_drop": orig_metrics["accuracy"] - shuffled_metrics["accuracy"],
        "nll_increase": shuffled_metrics["negative_log_likelihood"] - orig_metrics["negative_log_likelihood"],
        "context_dependence_confirmed": bool(shuffled_metrics["accuracy"] < 0.25),
    }

    out_file = rep_path / "exp_e1_shuffled_control.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(receipt, f, indent=2)

    print(f"\nSaved E1 receipt to {out_file}")
    print(f"Original Accuracy: {orig_metrics['accuracy']*100:.2f}%")
    print(f"Shuffled Context Accuracy: {shuffled_metrics['accuracy']*100:.2f}% (Drop: {receipt['accuracy_drop']*100:.2f}%)")
    print(f"Shuffled Abstention Rate: {shuffled_metrics['abstention_rate']*100:.2f}%")
    print(f"Original NLL: {orig_metrics['negative_log_likelihood']:.4f} -> Shuffled NLL: {shuffled_metrics['negative_log_likelihood']:.4f}")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description="Run E1 Shuffled-Context Control.")
    parser.add_argument("--test_file", type=str, default="data/real_banking_test.jsonl")
    parser.add_argument("--checkpoint_dir", type=str, default="artifacts/v2")
    parser.add_argument("--reports_dir", type=str, default="reports/v2")
    parser.add_argument("--model_name", type=str, default="knowledgator/gliclass-modern-base-v2.0")
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_shuffled_control(
        test_file=args.test_file,
        checkpoint_dir=args.checkpoint_dir,
        reports_dir=args.reports_dir,
        model_name=args.model_name,
        device_name=args.device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
