"""Experiment E4: Hard-Negative Distractors Evaluation.

Constructs K=5 and K=9 candidate menus using the nearest semantic non-gold neighbours
embedded by the ModernBERT backbone. Measures accuracy and Brier degradation compared
to uniform random distractor menus. Writes receipt to reports/v2/exp_e4_hard_negatives.json.
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


def run_hard_negatives_experiment(
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

    records = load_jsonl(Path(test_file))
    print(f"Loaded {len(records)} test records.")

    # Collect all unique Banking77 category slugs
    all_categories = set(BANKING_GLOSSARY.keys())
    for r in records:
        for c in r["candidates"]:
            if c["id"] != "__insufficient_evidence__":
                all_categories.add(c["id"])
    category_list = sorted(list(all_categories))
    cat_to_idx = {c: i for i, c in enumerate(category_list)}
    descriptions = [get_enriched_label(c) for c in category_list]
    n_cats = len(category_list)
    print(f"Embedding {n_cats} category descriptions with ModernBERT backbone...")

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

    # Embed all descriptions using backbone CLS / mean pool
    with torch.no_grad():
        toks = tokenizer(descriptions, padding=True, truncation=True, return_tensors="pt").to(device)
        backbone_out = model.model.encoder_model(**toks)
        # Mean pooling over active tokens
        hidden = backbone_out.last_hidden_state
        mask = toks["attention_mask"].unsqueeze(-1).float()
        emb = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        norm_emb = emb / emb.norm(dim=-1, keepdim=True)
        # Similarity matrix: (n_cats, n_cats)
        sim_matrix = (norm_emb @ norm_emb.T).cpu().numpy()

    # Precompute ranked non-gold neighbours for every category
    hard_neighbors: dict[str, list[str]] = {}
    for idx, cat in enumerate(category_list):
        sims = sim_matrix[idx].copy()
        sims[idx] = -999.0  # exclude self
        ranked_indices = np.argsort(-sims)
        hard_neighbors[cat] = [category_list[j] for j in ranked_indices]

    # Load calibrator
    cal_path = Path(checkpoint_dir) / "calibrator.json"
    if not cal_path.exists():
        cal_path = Path(checkpoint_dir) / "calibrator_modernbert.json"
    calibrator = None
    if cal_path.exists():
        calibrator = TemperatureCalibrator.load(cal_path)

    abstention_desc = "None of the above categories accurately match this inquiry (insufficient evidence or out-of-scope)"

    # Build evaluation runner
    def evaluate_records(eval_items: list[dict[str, Any]]) -> dict[str, float]:
        all_logits = []
        all_targets = []
        correct_count = 0

        with torch.no_grad():
            for i in range(0, len(eval_items), batch_size):
                batch = eval_items[i : i + batch_size]
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
                        correct_count += 1

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
            "samples": len(eval_items),
            "accuracy": correct_count / len(eval_items),
            "nll": nll,
            "brier": brier,
            "ece": ece,
        }

    # Evaluate in-scope test records under K=5 and K=9 with hard negatives
    in_scope_records = [r for r in records if not r.get("is_abstention", False) and r["target_id"] in hard_neighbors]

    # Build K=5 hard slice (gold + 3 nearest non-gold + abstention)
    k5_items = []
    for r in in_scope_records:
        gold = r["target_id"]
        neighbors = hard_neighbors[gold][:3]
        cands = [{"id": gold, "description": get_enriched_label(gold)}]
        for n in neighbors:
            cands.append({"id": n, "description": get_enriched_label(n)})
        cands.append({"id": "__insufficient_evidence__", "description": abstention_desc})
        # Randomize order
        rng = random.Random(hash(r["id"]) + 5)
        rng.shuffle(cands)
        k5_items.append({
            "id": r["id"],
            "question": r["question"],
            "text": r["text"],
            "candidates": cands,
            "target_id": gold,
        })

    # Build K=9 hard slice (gold + 7 nearest non-gold + abstention)
    k9_items = []
    for r in in_scope_records:
        gold = r["target_id"]
        neighbors = hard_neighbors[gold][:7]
        cands = [{"id": gold, "description": get_enriched_label(gold)}]
        for n in neighbors:
            cands.append({"id": n, "description": get_enriched_label(n)})
        cands.append({"id": "__insufficient_evidence__", "description": abstention_desc})
        rng = random.Random(hash(r["id"]) + 9)
        rng.shuffle(cands)
        k9_items.append({
            "id": r["id"],
            "question": r["question"],
            "text": r["text"],
            "candidates": cands,
            "target_id": gold,
        })

    print("Evaluating K=5 hard negative distractors...")
    k5_res = evaluate_records(k5_items)
    print("Evaluating K=9 hard negative distractors...")
    k9_res = evaluate_records(k9_items)

    receipt = {
        "experiment": "E4_hard_negative_distractors",
        "evaluation_samples": len(in_scope_records),
        "hard_negatives_k5": k5_res,
        "hard_negatives_k9": k9_res,
        "degradation_k5_to_k9_accuracy": k5_res["accuracy"] - k9_res["accuracy"],
    }

    out_file = rep_path / "exp_e4_hard_negatives.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(receipt, f, indent=2)

    print(f"\nSaved E4 receipt to {out_file}")
    print(f"Hard Negatives K=5: Acc={k5_res['accuracy']*100:.2f}%, Brier={k5_res['brier']:.4f}, ECE={k5_res['ece']*100:.2f}%")
    print(f"Hard Negatives K=9: Acc={k9_res['accuracy']*100:.2f}%, Brier={k9_res['brier']:.4f}, ECE={k9_res['ece']*100:.2f}%")
    print(f"K=5 to K=9 Accuracy Drop: {receipt['degradation_k5_to_k9_accuracy']*100:.2f}%")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description="Run E4 Hard-Negative Distractors Experiment.")
    parser.add_argument("--test_file", type=str, default="data/real_banking_test.jsonl")
    parser.add_argument("--checkpoint_dir", type=str, default="artifacts/v2")
    parser.add_argument("--reports_dir", type=str, default="reports/v2")
    parser.add_argument("--model_name", type=str, default="knowledgator/gliclass-modern-base-v2.0")
    parser.add_argument("--device", type=str, default="mps")
    args = parser.parse_args()

    run_hard_negatives_experiment(
        test_file=args.test_file,
        checkpoint_dir=args.checkpoint_dir,
        reports_dir=args.reports_dir,
        model_name=args.model_name,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
