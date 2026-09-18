"""Experiment E2: Option-Order Sensitivity and Permutation Invariance.

Evaluates test set predictions under:
1. Original candidate order
2. Reversed candidate order
3. Three distinct seeded shuffles

Measures:
- Argmax flip rate
- Mean total-variation distance (TVD)
- Flip correlation with prediction confidence
- Position bias of the explicit abstention candidate
Writes receipt to reports/v2/exp_e2_option_order.json.
"""

from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import AutoTokenizer

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


def run_option_order_experiment(
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
    n_records = len(records)
    print(f"Loaded {n_records} test records for option-order sensitivity evaluation.")

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

    # Load calibrator
    cal_path = Path(checkpoint_dir) / "calibrator.json"
    if not cal_path.exists():
        cal_path = Path(checkpoint_dir) / "calibrator_modernbert.json"
    calibrator = None
    if cal_path.exists():
        calibrator = TemperatureCalibrator.load(cal_path)

    # Order variations to test:
    # 0: original
    # 1: reversed
    # 2, 3, 4: seeded shuffles
    def get_permutations(cands: list[dict[str, str]], base_seed: int) -> list[tuple[str, list[dict[str, str]]]]:
        perms = [("original", list(cands)), ("reversed", list(reversed(cands)))]
        for s_idx in range(1, 4):
            shuffled = list(cands)
            rng = random.Random(base_seed + s_idx * 1000)
            rng.shuffle(shuffled)
            perms.append((f"shuffle_{s_idx}", shuffled))
        return perms

    # Dictionary: condition -> list of dicts with {id, pred_id, probs_by_id, top_conf, abstention_pos, abstention_prob}
    condition_results: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)

    conditions = ["original", "reversed", "shuffle_1", "shuffle_2", "shuffle_3"]

    for cond in conditions:
        print(f"Evaluating permutation condition: {cond}...")
        # Prepare evaluation items for this condition
        cond_items = []
        for idx, rec in enumerate(records):
            perms = get_permutations(rec["candidates"], seed + idx)
            perm_cands = next(p[1] for p in perms if p[0] == cond)
            cond_items.append(
                {
                    "id": rec["id"],
                    "question": rec["question"],
                    "text": rec["text"],
                    "candidates": perm_cands,
                    "target_id": rec["target_id"],
                }
            )

        with torch.no_grad():
            for i in range(0, len(cond_items), batch_size):
                batch = cond_items[i : i + batch_size]
                batch_labels = [[c["description"] for c in r["candidates"]] for r in batch]
                batch_ids = [[c["id"] for c in r["candidates"]] for r in batch]

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
                    top_conf = float(probs[pred_idx])

                    probs_by_id = {cid: float(probs[k]) for k, cid in enumerate(cand_ids)}

                    abs_pos = -1
                    abs_prob = 0.0
                    for k, cid in enumerate(cand_ids):
                        if cid == "__insufficient_evidence__":
                            abs_pos = k
                            abs_prob = float(probs[k])
                            break

                    condition_results[cond].append(
                        {
                            "id": batch[b_idx]["id"],
                            "pred_id": pred_id,
                            "top_conf": top_conf,
                            "probs_by_id": probs_by_id,
                            "abstention_pos": abs_pos,
                            "abstention_prob": abs_prob,
                        }
                    )

    # Compute comparison metrics against original
    orig_results = condition_results["original"]
    reversal_results = condition_results["reversed"]

    flips_by_condition = {}
    tv_distances_by_condition = {}

    for cond in ["reversed", "shuffle_1", "shuffle_2", "shuffle_3"]:
        comp_results = condition_results[cond]
        flips = 0
        tv_list = []
        for orig, comp in zip(orig_results, comp_results):
            if orig["pred_id"] != comp["pred_id"]:
                flips += 1
            # Total variation distance: 0.5 * sum(|p1 - p2|)
            all_ids = set(orig["probs_by_id"].keys()) | set(comp["probs_by_id"].keys())
            tv = 0.5 * sum(abs(orig["probs_by_id"].get(cid, 0.0) - comp["probs_by_id"].get(cid, 0.0)) for cid in all_ids)
            tv_list.append(tv)

        flips_by_condition[cond] = {
            "flips_count": flips,
            "flip_rate": flips / n_records,
            "mean_tv_distance": float(np.mean(tv_list)),
            "p95_tv_distance": float(np.percentile(tv_list, 95)),
        }

    # Analyze correlation between flip occurrence and confidence
    # Check for each record whether it flipped in ANY condition
    flipped_any = []
    confs_flipped = []
    confs_invariant = []

    for idx, orig in enumerate(orig_results):
        did_flip = any(
            condition_results[cond][idx]["pred_id"] != orig["pred_id"]
            for cond in ["reversed", "shuffle_1", "shuffle_2", "shuffle_3"]
        )
        flipped_any.append(did_flip)
        if did_flip:
            confs_flipped.append(orig["top_conf"])
        else:
            confs_invariant.append(orig["top_conf"])

    mean_conf_flipped = float(np.mean(confs_flipped)) if confs_flipped else 0.0
    mean_conf_invariant = float(np.mean(confs_invariant)) if confs_invariant else 0.0
    any_flip_rate = sum(flipped_any) / n_records

    # Abstention position analysis across all permutations
    # Group abstention probabilities by position index
    abs_probs_by_pos = collections.defaultdict(list)
    abs_selection_by_pos = collections.defaultdict(list)

    for cond in conditions:
        for res in condition_results[cond]:
            pos = res["abstention_pos"]
            if pos >= 0:
                abs_probs_by_pos[pos].append(res["abstention_prob"])
                abs_selection_by_pos[pos].append(1 if res["pred_id"] == "__insufficient_evidence__" else 0)

    pos_summary = {}
    for pos in sorted(abs_probs_by_pos.keys()):
        probs = abs_probs_by_pos[pos]
        selections = abs_selection_by_pos[pos]
        pos_summary[f"position_{pos}"] = {
            "samples": len(probs),
            "mean_abstention_prob": float(np.mean(probs)),
            "abstention_selection_rate": float(np.mean(selections)),
        }

    receipt = {
        "experiment": "E2_option_order_sensitivity",
        "sample_count": n_records,
        "seed": seed,
        "reversal_flip_rate": flips_by_condition["reversed"]["flip_rate"],
        "reversal_flips_count": flips_by_condition["reversed"]["flips_count"],
        "mean_tv_distance_reversal": flips_by_condition["reversed"]["mean_tv_distance"],
        "flips_by_condition": flips_by_condition,
        "any_flip_rate": any_flip_rate,
        "confidence_correlation": {
            "mean_conf_flipped": mean_conf_flipped,
            "mean_conf_invariant": mean_conf_invariant,
            "lower_confidence_drives_flips": bool(mean_conf_flipped < mean_conf_invariant),
        },
        "abstention_position_bias": pos_summary,
    }

    out_file = rep_path / "exp_e2_option_order.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(receipt, f, indent=2)

    print(f"\nSaved E2 receipt to {out_file}")
    print(f"Reversal Flip Rate: {receipt['reversal_flip_rate']*100:.2f}% ({receipt['reversal_flips_count']}/{n_records})")
    print(f"Any Permutation Flip Rate: {receipt['any_flip_rate']*100:.2f}%")
    print(f"Mean TV Distance (Reversal): {receipt['mean_tv_distance_reversal']:.4f}")
    print(f"Confidence of Invariant: {mean_conf_invariant*100:.1f}% vs Flipped: {mean_conf_flipped*100:.1f}%")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description="Run E2 Option-Order Sensitivity Experiment.")
    parser.add_argument("--test_file", type=str, default="data/real_banking_test.jsonl")
    parser.add_argument("--checkpoint_dir", type=str, default="artifacts/v2")
    parser.add_argument("--reports_dir", type=str, default="reports/v2")
    parser.add_argument("--model_name", type=str, default="knowledgator/gliclass-modern-base-v2.0")
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_option_order_experiment(
        test_file=args.test_file,
        checkpoint_dir=args.checkpoint_dir,
        reports_dir=args.reports_dir,
        model_name=args.model_name,
        device_name=args.device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
