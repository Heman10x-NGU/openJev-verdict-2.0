"""Experiment E3: Label-Length and Glossary-Coverage Bias.

Evaluates:
1. Accuracy split by whether gold label has a curated glossary entry or fallback template.
2. Correlation between candidate token length and predicted selection probability.
3. Model performance when all 77 categories are given curated descriptions in a uniform length band.

Writes receipt to reports/v2/exp_e3_label_bias.json.
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

from core.banking_glossary import BANKING_GLOSSARY, get_enriched_label
from core.calibration import TemperatureCalibrator
from core.formatting import build_model_input
from gliclass import GLiClassModel

# Uniform length curated descriptions for the 27 fallback categories (12-16 words)
ALL_77_UNIFORM_GLOSSARY: dict[str, str] = dict(BANKING_GLOSSARY)
FALLBACK_CURATED: dict[str, str] = {
    "activate_my_card": "Follow necessary security steps to activate a newly received physical debit or credit card",
    "age_limit": "Inquire about minimum and maximum age requirements to open and hold an active bank account",
    "apple_pay_or_google_pay": "Set up, troubleshoot, or use digital mobile wallet payment integration on smartphones and wearables",
    "card_acceptance": "Check whether bank issued debit cards are widely accepted at global merchant payment terminals",
    "card_payment_wrong_exchange_rate": "Dispute foreign currency exchange rate calculation applied to an international debit card purchase abroad",
    "country_support": "Check which international countries and sovereign jurisdictions are supported for banking services and residency",
    "direct_debit_payment_not_recognised": "Report an unfamiliar or unauthorized automated direct debit payment appearing on an account statement",
    "disposable_virtual_card": "Generate, manage, or dispose single-use temporary virtual cards for secure online merchant purchases",
    "getting_spare_card": "Request an additional backup physical card for travel, family members, or emergency redundancy",
    "getting_virtual_card": "Create and activate an instant digital virtual card for immediate online spending and payments",
    "lost_or_stolen_phone": "Report a lost or stolen mobile phone with banking app and security access credentials",
    "order_physical_card": "Order, reorder, or purchase a brand new embossed physical debit card for account spending",
    "passcode_forgotten": "Reset, recover, or unlock a forgotten mobile app passcode or online banking security login",
    "pending_card_payment": "Ask why an approved store or online merchant debit transaction remains in pending authorization",
    "receiving_money": "Inquire how to receive inbound domestic and international payments, bank transfers, or direct salary",
    "request_refund": "Ask how to initiate, request, or claim a formal merchant refund for purchased goods",
    "reverted_card_payment?": "Inquire why an existing debit card payment was unexpectedly cancelled, reversed, or refunded",
    "supported_cards_and_currencies": "Check which payment card networks and foreign fiat currencies are supported for account balances",
    "terminate_account": "Request permanent closure, formal cancellation, and final balance payout of an existing bank account",
    "top_up_by_bank_transfer_charge": "Ask about wire transfer fees or transaction surcharges when funding an account by bank transfer",
    "top_up_by_card_charge": "Inquire about debit card processing commissions and deposit fees when topping up account funds",
    "top_up_by_cash_or_cheque": "Ask about physical bank branch or cash deposit counter options for depositing money",
    "top_up_failed": "Troubleshoot why an attempted balance top-up or account deposit transaction failed or was rejected",
    "top_up_limits": "Inquire about rolling daily, weekly, and monthly deposit maximums and balance funding thresholds",
    "top_up_reverted": "Ask why a recently credited balance deposit or card top-up was reversed and deducted",
    "unable_to_verify_identity": "Resolve customer KYC identification document upload rejections and passport verification failures",
    "verify_my_identity": "Learn how to complete identity verification, upload government ID documents, or confirm home address",
    "virtual_card_not_working": "Troubleshoot reasons why a digital virtual card was declined or rejected during online checkout",
    "wrong_exchange_rate_for_cash_withdrawal": "Dispute foreign currency exchange rate conversions applied during an overseas ATM cash withdrawal",
}
ALL_77_UNIFORM_GLOSSARY.update(FALLBACK_CURATED)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def run_label_length_experiment(
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

    records = load_jsonl(Path(test_file))
    print(f"Loaded {len(records)} test records.")

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

    # 1. Standard evaluation tracking token lengths and curated vs templated accuracy
    curated_correct = 0
    curated_total = 0
    templated_correct = 0
    templated_total = 0

    all_token_lengths: list[int] = []
    all_assigned_probs: list[float] = []

    with torch.no_grad():
        for i in range(0, len(records), batch_size):
            batch = records[i : i + batch_size]
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
                target_id = target_ids[b_idx]
                target_idx = batch_ids[b_idx].index(target_id)
                is_correct = bool(pred_idx == target_idx)

                is_curated = target_id in BANKING_GLOSSARY
                if is_curated:
                    curated_total += 1
                    if is_correct:
                        curated_correct += 1
                else:
                    templated_total += 1
                    if is_correct:
                        templated_correct += 1

                for k, desc in enumerate(batch_labels[b_idx]):
                    tok_len = len(tokenizer.encode(desc, add_special_tokens=False))
                    all_token_lengths.append(tok_len)
                    all_assigned_probs.append(float(probs[k]))

    curated_acc = curated_correct / curated_total if curated_total > 0 else 0.0
    templated_acc = templated_correct / templated_total if templated_total > 0 else 0.0

    # Correlation between token length and selection probability
    lengths_arr = np.array(all_token_lengths, dtype=np.float64)
    probs_arr = np.array(all_assigned_probs, dtype=np.float64)

    # Pearson correlation
    pearson_corr = float(np.corrcoef(lengths_arr, probs_arr)[0, 1])

    # Spearman rank correlation
    rank_lengths = np.argsort(np.argsort(lengths_arr))
    rank_probs = np.argsort(np.argsort(probs_arr))
    spearman_corr = float(np.corrcoef(rank_lengths, rank_probs)[0, 1])

    # 2. Rerun with all 77 categories curated to uniform length band
    uniform_records = []
    for r in records:
        new_cands = []
        for c in r["candidates"]:
            cid = c["id"]
            if cid in ALL_77_UNIFORM_GLOSSARY:
                new_desc = ALL_77_UNIFORM_GLOSSARY[cid]
            else:
                new_desc = c["description"]
            new_cands.append({"id": cid, "description": new_desc})
        uniform_records.append(
            {
                "id": r["id"],
                "question": r["question"],
                "text": r["text"],
                "candidates": new_cands,
                "target_id": r["target_id"],
            }
        )

    uniform_correct = 0
    with torch.no_grad():
        for i in range(0, len(uniform_records), batch_size):
            batch = uniform_records[i : i + batch_size]
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
                    uniform_correct += 1

    uniform_acc = uniform_correct / len(uniform_records)

    receipt = {
        "experiment": "E3_label_length_and_glossary_bias",
        "total_test_records": len(records),
        "curated_categories_count": len(BANKING_GLOSSARY),
        "templated_categories_count": 77 - len(BANKING_GLOSSARY),
        "curated_gold_samples": curated_total,
        "templated_gold_samples": templated_total,
        "accuracy_by_glossary_status": {
            "curated_accuracy": curated_acc,
            "templated_accuracy": templated_acc,
            "gap": curated_acc - templated_acc,
        },
        "length_probability_correlation": {
            "pearson_r": pearson_corr,
            "spearman_rho": spearman_corr,
            "mean_token_length": float(np.mean(lengths_arr)),
            "std_token_length": float(np.std(lengths_arr)),
        },
        "uniform_glossary_rerun": {
            "uniform_accuracy": uniform_acc,
            "delta_from_standard": uniform_acc - (curated_correct + templated_correct) / len(records),
        },
    }

    out_file = rep_path / "exp_e3_label_bias.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(receipt, f, indent=2)

    print(f"\nSaved E3 receipt to {out_file}")
    print(f"Curated Label Accuracy:   {curated_acc*100:.2f}% ({curated_correct}/{curated_total})")
    print(f"Templated Label Accuracy: {templated_acc*100:.2f}% ({templated_correct}/{templated_total})")
    print(f"Glossary Gap:             {(curated_acc - templated_acc)*100:.2f}%")
    print(f"Token Length Correlation: Pearson r={pearson_corr:.4f}, Spearman rho={spearman_corr:.4f}")
    print(f"Uniform 77 Glossary Acc:  {uniform_acc*100:.2f}%")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description="Run E3 Label Length and Glossary Bias Experiment.")
    parser.add_argument("--test_file", type=str, default="data/real_banking_test.jsonl")
    parser.add_argument("--checkpoint_dir", type=str, default="artifacts/v2")
    parser.add_argument("--reports_dir", type=str, default="reports/v2")
    parser.add_argument("--model_name", type=str, default="knowledgator/gliclass-modern-base-v2.0")
    parser.add_argument("--device", type=str, default="mps")
    args = parser.parse_args()

    run_label_length_experiment(
        test_file=args.test_file,
        checkpoint_dir=args.checkpoint_dir,
        reports_dir=args.reports_dir,
        model_name=args.model_name,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
