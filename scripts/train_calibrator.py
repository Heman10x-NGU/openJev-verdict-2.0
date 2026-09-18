"""Calibrator training script using L-BFGS temperature scaling on held-out data.

Loads held-out calibration records, evaluates candidate logits with DecisionEngine,
fits a strictly positive scalar temperature T = exp(theta), and serializes the
calibrator artifact with verified ECE and Brier score metrics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from core.calibration import (
    TemperatureCalibrator,
    brier_score_loss,
    compute_ece,
    cross_entropy_loss,
)
from core.engine_encoder import DecisionEngine
from core.primitives import (
    INSUFFICIENT_EVIDENCE_DESC,
    INSUFFICIENT_EVIDENCE_ID,
    Choice,
    Option,
)


def load_dataset(file_path: str | Path) -> list[dict]:
    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit temperature calibrator")
    parser.add_argument(
        "--data_file",
        type=str,
        default="data/synthetic_calibration.jsonl",
        help="Path to calibration dataset",
    )
    parser.add_argument(
        "--output_artifact",
        type=str,
        default="artifacts/calibrator_modernbert.json",
        help="Output path for calibrator artifact",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="knowledgator/gliclass-modern-base-v2.0",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--mock", action="store_true", help="Use mock engine")
    args = parser.parse_args()

    print(f"Loading calibration data from {args.data_file}...")
    records = load_dataset(args.data_file)
    print(f"Loaded {len(records)} calibration records.")

    # Initialize DecisionEngine
    if args.mock:
        engine = DecisionEngine(model="mock")
    else:
        print("Initializing DecisionEngine with ModernBERT / GLiClass...")
        engine = DecisionEngine(model_name_or_path=args.model_name_or_path, device=args.device)

    # Standard routing query options
    categories = [
        ("billing", "Billing dispute, duplicate charge, or payment refund"),
        ("tech_support", "System crash, 500 error, or server degradation"),
        ("security", "Compromised API key, privilege escalation, or breach"),
        ("general", "General informational inquiry or SOC2 compliance"),
    ]
    options = [Option(id=cid, description=desc) for cid, desc in categories]
    option_ids = [cid for cid, _ in categories] + [INSUFFICIENT_EVIDENCE_ID]

    choice_query = Choice(
        id="routing_choice",
        question="Select the department to handle this inquiry",
        options=options,
    )

    all_logits = []
    all_targets = []

    print("Evaluating calibration samples...")
    for idx, r in enumerate(records):
        context = r["context"]
        target_id = r["choice_target"]

        # Run single evaluation
        res = engine.evaluate(context, [choice_query])
        c_res = res.results[0]

        # Extract ordered logits from probabilities (using log(p))
        # Or directly from model
        p_vec = [max(1e-12, c_res.probabilities[oid]) for oid in option_ids]
        logits_row = [float(np.log(p)) for p in p_vec]
        all_logits.append(logits_row)

        target_idx = option_ids.index(target_id)
        all_targets.append(target_idx)

    logits_tensor = torch.tensor(all_logits, dtype=torch.float32)
    targets_tensor = torch.tensor(all_targets, dtype=torch.long)

    # Compute uncalibrated metrics
    uncal_probs = F.softmax(logits_tensor, dim=-1).numpy()
    uncal_confs = np.max(uncal_probs, axis=-1)
    uncal_preds = np.argmax(uncal_probs, axis=-1)
    uncal_accs = (uncal_preds == targets_tensor.numpy()).astype(float)

    uncal_brier = brier_score_loss(logits_tensor, targets_tensor).item()
    uncal_nll = cross_entropy_loss(logits_tensor, targets_tensor).item()
    uncal_ece_width = compute_ece(uncal_confs, uncal_accs, n_bins=10, strategy="equal_width")
    uncal_ece_mass = compute_ece(uncal_confs, uncal_accs, n_bins=10, strategy="equal_mass")

    print("\n--- Uncalibrated Baseline ---")
    print(f"Accuracy: {np.mean(uncal_accs):.4f}")
    print(f"Brier Score: {uncal_brier:.4f}")
    print(f"NLL: {uncal_nll:.4f}")
    print(f"Equal-Width ECE: {uncal_ece_width.ece:.4f} (MCE: {uncal_ece_width.mce:.4f})")
    print(f"Equal-Mass ECE:  {uncal_ece_mass.ece:.4f} (MCE: {uncal_ece_mass.mce:.4f})")

    # Fit calibrator
    print("\nFitting TemperatureCalibrator via L-BFGS...")
    calibrator = TemperatureCalibrator(model_id=args.model_name_or_path)
    calibrator.fit(logits_tensor, targets_tensor, max_iter=50)

    fitted_temp = calibrator.temperature
    print(f"Optimal Temperature: {fitted_temp:.4f}")

    # Compute calibrated metrics
    cal_logits = calibrator(logits_tensor)
    cal_probs = F.softmax(cal_logits, dim=-1).detach().numpy()
    cal_confs = np.max(cal_probs, axis=-1)
    cal_preds = np.argmax(cal_probs, axis=-1)
    cal_accs = (cal_preds == targets_tensor.numpy()).astype(float)

    cal_brier = brier_score_loss(cal_logits, targets_tensor).item()
    cal_nll = cross_entropy_loss(cal_logits, targets_tensor).item()
    cal_ece_width = compute_ece(cal_confs, cal_accs, n_bins=10, strategy="equal_width")
    cal_ece_mass = compute_ece(cal_confs, cal_accs, n_bins=10, strategy="equal_mass")

    print("\n--- Calibrated Results ---")
    print(f"Accuracy: {np.mean(cal_accs):.4f} (strictly preserved)")
    print(f"Brier Score: {cal_brier:.4f}")
    print(f"NLL: {cal_nll:.4f}")
    print(f"Equal-Width ECE: {cal_ece_width.ece:.4f} (MCE: {cal_ece_width.mce:.4f})")
    print(f"Equal-Mass ECE:  {cal_ece_mass.ece:.4f} (MCE: {cal_ece_mass.mce:.4f})")

    # Save artifact
    out_artifact = Path(args.output_artifact)
    artifact_data = {
        **calibrator.to_dict(),
        "uncalibrated_metrics": {
            "brier": uncal_brier,
            "nll": uncal_nll,
            "ece_equal_width": uncal_ece_width.ece,
            "ece_equal_mass": uncal_ece_mass.ece,
        },
        "calibrated_metrics": {
            "brier": cal_brier,
            "nll": cal_nll,
            "ece_equal_width": cal_ece_width.ece,
            "ece_equal_mass": cal_ece_mass.ece,
        },
        "calibration_samples_count": len(records),
    }

    out_artifact.parent.mkdir(parents=True, exist_ok=True)
    with open(out_artifact, "w", encoding="utf-8") as f:
        json.dump(artifact_data, f, indent=2)

    print(f"\nSaved calibration artifact to {out_artifact}")


if __name__ == "__main__":
    main()
