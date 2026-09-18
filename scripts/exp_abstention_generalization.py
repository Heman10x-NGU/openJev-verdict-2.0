"""Experiment E5: Abstention Generalisation Audit.

Evaluates abstention robustness beyond in-distribution training augmentation:
1. Standard baseline missing-option slice (K=5 uniform distractors, standard phrasing).
2. Hard-negative missing-option slice (K=9 nearest non-gold distractors).
3. Held-out synonym abstention phrasing (semantic paraphrase of abstention option).
4. Combined hard-negative distractors (K=9) + held-out synonym phrasing.

Writes receipt to reports/v2/exp_e5_abstention_generalization.json.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import AutoTokenizer

from core.banking_glossary import BANKING_GLOSSARY, get_enriched_label
from core.calibration import TemperatureCalibrator
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


SYNONYM_ABSTENTIONS = [
    "None of the above options accurately categorize this customer request",
    "This inquiry falls outside all available categories listed above (abstain)",
    "Insufficient evidence to match any of the provided choices",
]


def run_abstention_generalization_experiment(
    missing_option_file: str = "data/slice_missing_option.jsonl",
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

    base_slice = load_jsonl(Path(missing_option_file))
    print(f"Loaded {len(base_slice)} standard missing-option records.")

    # Load model
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

    # Precompute nearest neighbors across all 77 categories
    all_categories = sorted(list(BANKING_GLOSSARY.keys()))
    descriptions = [get_enriched_label(c) for c in all_categories]
    with torch.no_grad():
        toks = tokenizer(descriptions, padding=True, truncation=True, return_tensors="pt").to(device)
        out = model.model.encoder_model(**toks)
        hidden = out.last_hidden_state
        mask = toks["attention_mask"].unsqueeze(-1).float()
        emb = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        norm_emb = emb / emb.norm(dim=-1, keepdim=True)
        sim_matrix = (norm_emb @ norm_emb.T).cpu().numpy()

    hard_neighbors: dict[str, list[str]] = {}
    for idx, cat in enumerate(all_categories):
        sims = sim_matrix[idx].copy()
        sims[idx] = -999.0
        ranked = np.argsort(-sims)
        hard_neighbors[cat] = [all_categories[j] for j in ranked]

    # Load calibrator
    cal_path = Path(checkpoint_dir) / "calibrator.json"
    if not cal_path.exists():
        cal_path = Path(checkpoint_dir) / "calibrator_modernbert.json"
    calibrator = None
    if cal_path.exists():
        calibrator = TemperatureCalibrator.load(cal_path)

    def evaluate_slice(slice_records: list[dict[str, Any]]) -> dict[str, float]:
        tp = 0
        fp = 0
        fn = 0
        total = len(slice_records)

        with torch.no_grad():
            for i in range(0, total, batch_size):
                batch = slice_records[i : i + batch_size]
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
                    cand_ids = batch_ids[b_idx]
                    pred_id = cand_ids[pred_idx]
                    gold_id = target_ids[b_idx]

                    is_pred_abs = pred_id == "__insufficient_evidence__"
                    is_gold_abs = gold_id == "__insufficient_evidence__"

                    if is_pred_abs and is_gold_abs:
                        tp += 1
                    elif is_pred_abs and not is_gold_abs:
                        fp += 1
                    elif not is_pred_abs and is_gold_abs:
                        fn += 1

        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        return {
            "samples": total,
            "abstention_recall": rec,
            "abstention_precision": prec,
            "abstention_f1": f1,
            "true_abstentions": tp,
            "false_positives": fp,
            "false_negatives": fn,
        }

    # Slice 1: Standard Missing-Option (baseline)
    print("Evaluating Condition 1: Baseline Missing-Option...")
    res_baseline = evaluate_slice(base_slice)

    # Slice 2: Hard-Negative Missing-Option (K=9)
    # Rebuild missing-option items with 8 nearest neighbors of the omitted intent
    print("Evaluating Condition 2: Hard-Negative Missing-Option (K=9)...")
    hard_k9_slice = []
    for r in base_slice:
        # Original ground truth before omission is stored in r['original_target_id'] or inferred
        orig_intent = r.get("original_target_id", r["candidates"][0]["id"])
        neighbors = hard_neighbors.get(orig_intent, list(BANKING_GLOSSARY.keys()))[:8]
        new_cands = [{"id": n, "description": get_enriched_label(n)} for n in neighbors]
        new_cands.append(
            {
                "id": "__insufficient_evidence__",
                "description": "None of the above categories accurately match this inquiry (insufficient evidence or out-of-scope)",
            }
        )
        rng = random.Random(hash(r["id"]) + 99)
        rng.shuffle(new_cands)
        hard_k9_slice.append(
            {
                "id": r["id"],
                "question": r["question"],
                "text": r["text"],
                "candidates": new_cands,
                "target_id": "__insufficient_evidence__",
            }
        )
    res_hard_k9 = evaluate_slice(hard_k9_slice)

    # Slice 3: Held-Out Synonym Phrasing
    print("Evaluating Condition 3: Held-Out Synonym Phrasing...")
    synonym_slice = []
    for idx, r in enumerate(base_slice):
        chosen_synonym = SYNONYM_ABSTENTIONS[idx % len(SYNONYM_ABSTENTIONS)]
        new_cands = []
        for c in r["candidates"]:
            if c["id"] == "__insufficient_evidence__":
                new_cands.append({"id": "__insufficient_evidence__", "description": chosen_synonym})
            else:
                new_cands.append(dict(c))
        synonym_slice.append(
            {
                "id": r["id"],
                "question": r["question"],
                "text": r["text"],
                "candidates": new_cands,
                "target_id": "__insufficient_evidence__",
            }
        )
    res_synonym = evaluate_slice(synonym_slice)

    # Slice 4: Combined Hard-Negatives (K=9) + Synonym Phrasing
    print("Evaluating Condition 4: Combined Hard-Negatives (K=9) + Synonym Phrasing...")
    combined_slice = []
    for idx, r in enumerate(hard_k9_slice):
        chosen_synonym = SYNONYM_ABSTENTIONS[idx % len(SYNONYM_ABSTENTIONS)]
        new_cands = []
        for c in r["candidates"]:
            if c["id"] == "__insufficient_evidence__":
                new_cands.append({"id": "__insufficient_evidence__", "description": chosen_synonym})
            else:
                new_cands.append(dict(c))
        combined_slice.append(
            {
                "id": r["id"],
                "question": r["question"],
                "text": r["text"],
                "candidates": new_cands,
                "target_id": "__insufficient_evidence__",
            }
        )
    res_combined = evaluate_slice(combined_slice)

    receipt = {
        "experiment": "E5_abstention_generalization",
        "sample_count": len(base_slice),
        "baseline_standard_recall": res_baseline["abstention_recall"],
        "baseline_standard": res_baseline,
        "hard_negatives_k9": res_hard_k9,
        "synonym_phrasing": res_synonym,
        "combined_hard_and_synonym": res_combined,
        "recall_delta_hard_k9": res_hard_k9["abstention_recall"] - res_baseline["abstention_recall"],
        "recall_delta_synonym": res_synonym["abstention_recall"] - res_baseline["abstention_recall"],
        "recall_delta_combined": res_combined["abstention_recall"] - res_baseline["abstention_recall"],
    }

    out_file = rep_path / "exp_e5_abstention_generalization.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(receipt, f, indent=2)

    print(f"\nSaved E5 receipt to {out_file}")
    print(f"Baseline Standard Recall:           {res_baseline['abstention_recall']*100:.2f}%")
    print(f"Hard-Negatives (K=9) Recall:         {res_hard_k9['abstention_recall']*100:.2f}%")
    print(f"Synonym Phrasing Recall:             {res_synonym['abstention_recall']*100:.2f}%")
    print(f"Combined (Hard K=9 + Synonym) Recall:{res_combined['abstention_recall']*100:.2f}%")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description="Run E5 Abstention Generalization Experiment.")
    parser.add_argument("--missing_option_file", type=str, default="data/slice_missing_option.jsonl")
    parser.add_argument("--test_file", type=str, default="data/real_banking_test.jsonl")
    parser.add_argument("--checkpoint_dir", type=str, default="artifacts/v2")
    parser.add_argument("--reports_dir", type=str, default="reports/v2")
    parser.add_argument("--model_name", type=str, default="knowledgator/gliclass-modern-base-v2.0")
    parser.add_argument("--device", type=str, default="mps")
    args = parser.parse_args()

    run_abstention_generalization_experiment(
        missing_option_file=args.missing_option_file,
        test_file=args.test_file,
        checkpoint_dir=args.checkpoint_dir,
        reports_dir=args.reports_dir,
        model_name=args.model_name,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
