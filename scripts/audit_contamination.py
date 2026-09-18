"""Experiment E6: Character 5-Gram Contamination and Paraphrase Leakage Audit.

Computes character 5-gram Jaccard similarity across all train-test pairs in PolyAI Banking77.
Reports counts at similarity thresholds 0.7, 0.8, 0.9, 1.0, records the top 20 highest pairs
verbatim, filters the test set to remove pairs with similarity >= 0.8, and evaluates the
performance delta on the strictly decontaminated subset.

Writes receipt to reports/v2/exp_e6_contamination.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import AutoTokenizer

from core.calibration import TemperatureCalibrator, brier_score_loss, compute_ece, cross_entropy_loss
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


def get_5grams(text: str) -> set[str]:
    cleaned = " ".join(text.lower().split())
    if len(cleaned) < 5:
        return {cleaned}
    return {cleaned[i : i + 5] for i in range(len(cleaned) - 4)}


def run_contamination_audit(
    train_file: str = "data/real_banking_train.jsonl",
    test_file: str = "data/real_banking_test.jsonl",
    checkpoint_dir: str = "artifacts/v2",
    reports_dir: str = "reports/v2",
    model_name: str = "knowledgator/gliclass-modern-base-v2.0",
    device_name: str = "mps",
    batch_size: int = 16,
) -> dict[str, Any]:
    rep_path = Path(reports_dir)
    rep_path.mkdir(parents=True, exist_ok=True)

    if device_name == "mps" and not torch.backends.mps.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(device_name)

    print(f"Loading train ({train_file}) and test ({test_file}) sets...")
    train_records = load_jsonl(Path(train_file))
    test_records = load_jsonl(Path(test_file))

    train_grams = [(r["id"], r["text"], get_5grams(r["text"])) for r in train_records]
    test_grams = [(r["id"], r["text"], get_5grams(r["text"])) for r in test_records]

    print(f"Auditing {len(test_records)} test samples against {len(train_records)} train samples (2.3M pairs)...")

    # Inverted index from 5-gram to list of train indices to accelerate search
    ngram_index: dict[str, list[int]] = {}
    for t_idx, (_, _, grams) in enumerate(train_grams):
        for g in grams:
            ngram_index.setdefault(g, []).append(t_idx)

    max_jaccards_per_test: list[float] = []
    top_pairs: list[dict[str, Any]] = []

    count_ge_07 = 0
    count_ge_08 = 0
    count_ge_09 = 0
    count_eq_10 = 0

    decontaminated_test_records = []

    for test_idx, (t_id, t_text, t_g) in enumerate(test_grams):
        # Gather candidate train items sharing at least one 5-gram
        candidate_train_indices: set[int] = set()
        for g in t_g:
            if g in ngram_index:
                candidate_train_indices.update(ngram_index[g])

        max_j = 0.0
        best_train_item = None

        len_tg = len(t_g)
        for tr_idx in candidate_train_indices:
            tr_id, tr_text, tr_g = train_grams[tr_idx]
            intersection_len = len(t_g.intersection(tr_g))
            union_len = len_tg + len(tr_g) - intersection_len
            jaccard = intersection_len / union_len if union_len > 0 else 0.0

            if jaccard > max_j:
                max_j = jaccard
                best_train_item = (tr_id, tr_text)

        max_jaccards_per_test.append(max_j)

        if max_j >= 0.7:
            count_ge_07 += 1
        if max_j >= 0.8:
            count_ge_08 += 1
        if max_j >= 0.9:
            count_ge_09 += 1
        if max_j >= 0.9999:
            count_eq_10 += 1

        if best_train_item is not None and max_j >= 0.5:
            top_pairs.append(
                {
                    "test_id": t_id,
                    "test_text": t_text,
                    "train_id": best_train_item[0],
                    "train_text": best_train_item[1],
                    "jaccard_5gram": float(max_j),
                }
            )

        if max_j < 0.8:
            decontaminated_test_records.append(test_records[test_idx])

    top_pairs.sort(key=lambda x: x["jaccard_5gram"], reverse=True)
    top_20_verbatim = top_pairs[:20]

    print(f"Pairs with 5-gram Jaccard >= 0.7: {count_ge_07}")
    print(f"Pairs with 5-gram Jaccard >= 0.8: {count_ge_08}")
    print(f"Pairs with 5-gram Jaccard >= 0.9: {count_ge_09}")
    print(f"Pairs with 5-gram Jaccard == 1.0: {count_eq_10}")
    print(f"Decontaminated test set size (J < 0.8): {len(decontaminated_test_records)}/{len(test_records)}")

    # Load model and evaluate both full and decontaminated test sets
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

    cal_path = Path(checkpoint_dir) / "calibrator.json"
    if not cal_path.exists():
        cal_path = Path(checkpoint_dir) / "calibrator_modernbert.json"
    calibrator = None
    if cal_path.exists():
        calibrator = TemperatureCalibrator.load(cal_path)

    def evaluate_dataset(eval_records: list[dict[str, Any]]) -> dict[str, float]:
        all_logits = []
        all_targets = []
        correct = 0

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
                        correct += 1

                    all_logits.append(c_logits.cpu())
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
            "samples": len(eval_records),
            "accuracy": correct / len(eval_records),
            "nll": nll,
            "brier": brier,
            "ece": ece,
        }

    print("Evaluating full test set...")
    full_res = evaluate_dataset(test_records)
    print("Evaluating decontaminated test set (J < 0.8)...")
    decontam_res = evaluate_dataset(decontaminated_test_records)

    receipt = {
        "experiment": "E6_contamination_audit",
        "methodology": "character_5gram_jaccard_similarity",
        "train_samples": len(train_records),
        "test_samples": len(test_records),
        "threshold_counts": {
            "ge_0_7": count_ge_07,
            "ge_0_8": count_ge_08,
            "ge_0_9": count_ge_09,
            "exact_match_1_0": count_eq_10,
        },
        "top_20_paraphrase_pairs": top_20_verbatim,
        "full_test_metrics": full_res,
        "decontaminated_test_metrics": decontam_res,
        "delta": {
            "accuracy": decontam_res["accuracy"] - full_res["accuracy"],
            "brier": decontam_res["brier"] - full_res["brier"],
            "ece": decontam_res["ece"] - full_res["ece"],
        },
    }

    out_file = rep_path / "exp_e6_contamination.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(receipt, f, indent=2)

    print(f"\nSaved E6 receipt to {out_file}")
    print(f"Full Test Accuracy:           {full_res['accuracy']*100:.2f}% (ECE: {full_res['ece']*100:.2f}%)")
    print(f"Decontaminated Test Accuracy: {decontam_res['accuracy']*100:.2f}% (ECE: {decontam_res['ece']*100:.2f}%)")
    print(f"Delta:                        {receipt['delta']['accuracy']*100:+.2f}%")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description="Run E6 Character 5-gram Contamination Audit.")
    parser.add_argument("--train_file", type=str, default="data/real_banking_train.jsonl")
    parser.add_argument("--test_file", type=str, default="data/real_banking_test.jsonl")
    parser.add_argument("--checkpoint_dir", type=str, default="artifacts/v2")
    parser.add_argument("--reports_dir", type=str, default="reports/v2")
    parser.add_argument("--model_name", type=str, default="knowledgator/gliclass-modern-base-v2.0")
    parser.add_argument("--device", type=str, default="mps")
    args = parser.parse_args()

    run_contamination_audit(
        train_file=args.train_file,
        test_file=args.test_file,
        checkpoint_dir=args.checkpoint_dir,
        reports_dir=args.reports_dir,
        model_name=args.model_name,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
