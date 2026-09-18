"""Fine-tune Verdict-open-jev-ModernBERT with composite Cross-Entropy + Brier score loss.

Fine-tunes ModernBERT backbone and candidate classification head on enterprise
decisions with explicit abstention candidates. Runs on Apple Silicon MPS or CPU
with gradient accumulation, FP32 proper scoring loss, and safetensors checkpointing.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from core.calibration import brier_score_loss, composite_loss, cross_entropy_loss
from core.formatting import build_model_input, format_prompt
from gliclass import GLiClassModel


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load JSONL dataset."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def evaluate(
    model: GLiClassModel,
    tokenizer: Any,
    records: list[dict[str, Any]],
    device: torch.device,
    batch_size: int = 16,
) -> dict[str, float]:
    """Evaluate accuracy, cross-entropy, and Brier score on validation records."""
    model.eval()
    total_samples = len(records)
    total_correct = 0
    total_brier = 0.0
    total_nll = 0.0
    abstention_correct = 0
    total_abstentions = 0

    with torch.no_grad():
        for i in range(0, total_samples, batch_size):
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
            raw_logits = outputs.logits  # shape: (B, max_classes)

            for b_idx in range(len(batch)):
                labels = batch_labels[b_idx]
                k_classes = len(labels)
                c_logits = raw_logits[b_idx, :k_classes].float().unsqueeze(0)  # (1, K)
                c_ids = batch_ids[b_idx]
                target_id = target_ids[b_idx]

                target_idx = c_ids.index(target_id)
                target_tensor = torch.tensor([target_idx], dtype=torch.long, device=device)

                probs = F.softmax(c_logits, dim=-1)
                pred_idx = int(torch.argmax(probs, dim=-1).item())

                if pred_idx == target_idx:
                    total_correct += 1

                if batch[b_idx]["is_abstention"]:
                    total_abstentions += 1
                    if pred_idx == target_idx:
                        abstention_correct += 1

                nll = cross_entropy_loss(c_logits, target_tensor).item()
                brier = brier_score_loss(c_logits, target_tensor).item()
                total_nll += nll
                total_brier += brier

    acc = total_correct / total_samples if total_samples > 0 else 0.0
    mean_nll = total_nll / total_samples if total_samples > 0 else 0.0
    mean_brier = total_brier / total_samples if total_samples > 0 else 0.0
    abs_recall = abstention_correct / total_abstentions if total_abstentions > 0 else 0.0

    return {
        "accuracy": acc,
        "nll": mean_nll,
        "brier": mean_brier,
        "abstention_recall": abs_recall,
    }


def train(
    train_file: str,
    val_file: str,
    output_dir: str,
    model_name: str = "knowledgator/gliclass-modern-base-v2.0",
    epochs: int = 3,
    batch_size: int = 8,
    grad_accum_steps: int = 4,
    backbone_lr: float = 2e-5,
    head_lr: float = 1e-4,
    lambda_brier: float = 1.0,
    device_name: str = "mps",
    max_samples: int | None = None,
    seed: int = 42,
    selection_metric: str = "val_nll",
    warmup_ratio: float = 0.1,
) -> None:
    """Train Verdict-open-jev-ModernBERT with proper scoring rule composite loss."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if device_name == "mps" and not torch.backends.mps.is_available():
        print("MPS requested but not available. Falling back to CPU.")
        device = torch.device("cpu")
    elif device_name == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(device_name)

    print(f"Training device: {device.type.upper()}")
    print(f"Loading base model: {model_name}...")

    model = GLiClassModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.to(device)

    # Load dataset
    print(f"Loading data from {train_file} and {val_file}...")
    train_records = load_jsonl(Path(train_file))
    val_records = load_jsonl(Path(val_file))

    if max_samples is not None:
        train_records = train_records[:max_samples]
        val_records = val_records[: min(len(val_records), max(10, max_samples // 4))]

    print(f"Train samples: {len(train_records)}, Val samples: {len(val_records)}")

    # Differential learning rate parameter groups
    head_params = []
    backbone_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "classifier" in name or "scorer" in name or "classification" in name:
            head_params.append(param)
        else:
            backbone_params.append(param)

    optimizer_grouped_parameters = [
        {"params": backbone_params, "lr": backbone_lr, "weight_decay": 0.01},
        {"params": head_params, "lr": head_lr, "weight_decay": 0.01},
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters)

    # Initial evaluation
    print("Evaluating initial zero-shot baseline...")
    init_val_metrics = evaluate(model, tokenizer, val_records, device=device)
    print(
        f"Baseline -> Acc: {init_val_metrics['accuracy']:.4f}, "
        f"NLL: {init_val_metrics['nll']:.4f}, "
        f"Brier: {init_val_metrics['brier']:.4f}, "
        f"Abstention Recall: {init_val_metrics['abstention_recall']:.4f}"
    )

    norm_metric = selection_metric.lower().removeprefix("val_")
    valid_metrics = ("nll", "brier", "accuracy", "abstention_recall")
    if norm_metric not in valid_metrics:
        raise ValueError(
            f"Unknown selection metric '{selection_metric}'. "
            f"Valid options: {valid_metrics}"
        )
    is_lower_better = norm_metric in ("nll", "brier")
    best_metric_val = float("inf") if is_lower_better else float("-inf")
    best_epoch = 1
    best_val_accuracy = init_val_metrics["accuracy"]
    best_val_brier = init_val_metrics["brier"]
    best_val_nll = init_val_metrics["nll"]
    best_val_abstention_recall = init_val_metrics["abstention_recall"]
    training_history = []

    num_batches_per_epoch = len(range(0, len(train_records), batch_size))
    num_opt_steps_per_epoch = (num_batches_per_epoch + grad_accum_steps - 1) // grad_accum_steps
    total_opt_steps = num_opt_steps_per_epoch * epochs
    num_warmup_steps = int(total_opt_steps * warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_opt_steps,
    )

    total_steps = num_batches_per_epoch * epochs
    print(
        f"Starting training: {epochs} epochs, {total_steps} micro-steps, "
        f"{total_opt_steps} optimizer steps ({num_warmup_steps} warmup steps)..."
    )

    step = 0
    start_train_time = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()
        epoch_start = time.perf_counter()

        random.shuffle(train_records)

        for i in range(0, len(train_records), batch_size):
            batch = train_records[i : i + batch_size]
            if not batch:
                continue

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
            raw_logits = outputs.logits  # shape: (B, max_classes)

            # Compute composite loss across batch items
            batch_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
            for b_idx in range(len(batch)):
                k_classes = len(batch_labels[b_idx])
                c_logits = raw_logits[b_idx, :k_classes].float().unsqueeze(0)
                c_ids = batch_ids[b_idx]
                target_id = target_ids[b_idx]
                target_idx = c_ids.index(target_id)
                target_tensor = torch.tensor(
                    [target_idx], dtype=torch.long, device=device
                )

                item_loss = composite_loss(
                    c_logits, target_tensor, lambda_brier=lambda_brier
                )
                batch_loss = batch_loss + item_loss

            batch_loss = batch_loss / (len(batch) * grad_accum_steps)
            batch_loss.backward()

            step += 1
            epoch_loss += batch_loss.item() * grad_accum_steps

            if step % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if step % (grad_accum_steps * 20) == 0:
                print(
                    f"Epoch {epoch}/{epochs} | Step {step}/{total_steps} | "
                    f"Batch Loss: {batch_loss.item() * grad_accum_steps:.4f}"
                )

        if step % grad_accum_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        epoch_duration = time.perf_counter() - epoch_start
        print(f"Epoch {epoch} completed in {epoch_duration:.1f}s. Evaluating...")

        val_metrics = evaluate(model, tokenizer, val_records, device=device)
        print(
            f"Epoch {epoch} Validation -> "
            f"Acc: {val_metrics['accuracy']:.4f}, "
            f"NLL: {val_metrics['nll']:.4f}, "
            f"Brier: {val_metrics['brier']:.4f}, "
            f"Abstention Recall: {val_metrics['abstention_recall']:.4f}"
        )

        training_history.append(
            {
                "epoch": epoch,
                "loss": epoch_loss / max(1, len(train_records) // batch_size),
                "duration_seconds": epoch_duration,
                "val_accuracy": val_metrics["accuracy"],
                "val_nll": val_metrics["nll"],
                "val_brier": val_metrics["brier"],
                "val_abstention_recall": val_metrics["abstention_recall"],
            }
        )

        # Save checkpoint for this epoch under epoch_{n}/
        epoch_dir = out_path / f"epoch_{epoch}"
        epoch_dir.mkdir(parents=True, exist_ok=True)
        state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        save_file(state_dict, str(epoch_dir / "openjev_modernbert.safetensors"))
        save_file(state_dict, str(epoch_dir / "model.safetensors"))
        torch.save(state_dict, str(epoch_dir / "openjev_modernbert.pt"))
        tokenizer.save_pretrained(str(epoch_dir))
        model.config.save_pretrained(str(epoch_dir))
        with open(epoch_dir / "val_metrics.json", "w", encoding="utf-8") as f:
            json.dump(val_metrics, f, indent=2)

        current_val = val_metrics[norm_metric]
        is_best = (current_val < best_metric_val) if is_lower_better else (current_val > best_metric_val)

        if is_best:
            best_metric_val = current_val
            best_epoch = epoch
            best_val_accuracy = val_metrics["accuracy"]
            best_val_brier = val_metrics["brier"]
            best_val_nll = val_metrics["nll"]
            best_val_abstention_recall = val_metrics["abstention_recall"]
            print(
                f"New best {selection_metric} ({best_metric_val:.4f}) at epoch {epoch}! "
                f"Updating primary checkpoint in {out_path}..."
            )

            save_file(state_dict, str(out_path / "openjev_modernbert.safetensors"))
            save_file(state_dict, str(out_path / "model.safetensors"))
            torch.save(state_dict, str(out_path / "openjev_modernbert.pt"))
            tokenizer.save_pretrained(str(out_path))
            model.config.save_pretrained(str(out_path))

    total_duration = time.perf_counter() - start_train_time
    print(
        f"Training finished in {total_duration:.1f}s! "
        f"Selected epoch {best_epoch} based on {selection_metric} ({best_metric_val:.4f})."
    )

    manifest = {
        "model_architecture": "ModernBERT-base + GLiClass-v2",
        "parameters": sum(p.numel() for p in model.parameters()),
        "epochs": epochs,
        "batch_size": batch_size,
        "effective_batch_size": batch_size * grad_accum_steps,
        "loss_function": "CE + 1.0 * Brier (Proper Scoring Rule)",
        "selection_metric": selection_metric,
        "selected_epoch": best_epoch,
        "scheduler": "get_cosine_schedule_with_warmup",
        "warmup_ratio": warmup_ratio,
        "best_val_accuracy": best_val_accuracy,
        "best_val_brier": best_val_brier,
        "best_val_nll": best_val_nll,
        "best_val_abstention_recall": best_val_abstention_recall,
        "total_training_seconds": total_duration,
        "device": device.type,
        "seed": seed,
        "history": training_history,
    }
    with open(out_path / "train_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Training manifest saved to {out_path / 'train_manifest.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune Verdict-open-jev-ModernBERT.")
    parser.add_argument("--train_file", type=str, default="data/real_banking_train.jsonl")
    parser.add_argument("--val_file", type=str, default="data/real_banking_val.jsonl")
    parser.add_argument("--output_dir", type=str, default="artifacts")
    parser.add_argument("--model_name", type=str, default="knowledgator/gliclass-modern-base-v2.0")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--selection_metric",
        type=str,
        default="val_nll",
        choices=["val_nll", "val_brier", "val_accuracy", "val_abstention_recall"],
        help="Validation metric for checkpoint selection (default: val_nll)",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.1,
        help="Ratio of warmup steps for cosine decay schedule (default: 0.1)",
    )
    args = parser.parse_args()

    train(
        train_file=args.train_file,
        val_file=args.val_file,
        output_dir=args.output_dir,
        model_name=args.model_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        device_name=args.device,
        max_samples=args.max_samples,
        seed=args.seed,
        selection_metric=args.selection_metric,
        warmup_ratio=args.warmup_ratio,
    )


if __name__ == "__main__":
    main()
