#!/usr/bin/env python3
"""Train Verdict 2.0, fit calibration, and report dev metrics.

The test split is never read here. Use verdict2/evaluate.py for the single final test read.

Example:
    python -m verdict2.train --backbone answerdotai/ModernBERT-base --epochs 6 --limit 40   # smoke test
    python -m verdict2.train --backbone answerdotai/ModernBERT-large --epochs 8             # full run
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from .data import Item, collate, load_items, split_by_case
from .losses import decision_loss
from .metrics import summarize
from .model import CorrectnessHead, VerdictModel, apply_temperature


def pick_device(requested: str | None) -> str:
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def batches(items: List[Item], size: int, pad_id: int, shuffle: bool, rng: random.Random):
    order = list(range(len(items)))
    if shuffle:
        rng.shuffle(order)
    # length bucketing keeps padding low, which matters most on a 6 GB card
    order.sort(key=lambda i: len(items[i].ids) // 64)
    chunks = [order[i : i + size] for i in range(0, len(order), size)]
    if shuffle:
        rng.shuffle(chunks)
    for chunk in chunks:
        picked = [items[i] for i in chunk]
        yield collate(picked, pad_id), picked


def to_device(batch: Dict[str, torch.Tensor], device: str) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


@torch.no_grad()
def infer(model: VerdictModel, items: List[Item], pad_id: int, device: str, batch_size: int, use_temperature: bool) -> List[Dict]:
    """Run the model over items and return per-decision records for metrics."""
    model.eval()
    records: List[Dict] = []
    rng = random.Random(0)
    for batch, chunk in batches(items, batch_size, pad_id, shuffle=False, rng=rng):
        dev_batch = to_device(batch, device)
        logits = model.option_logits(dev_batch)
        k = dev_batch["marker_mask"].sum(-1)
        if use_temperature:
            logits = apply_temperature(logits, dev_batch["qtype"], k, model.temperature)
        probs = torch.softmax(logits, dim=-1)
        feats = CorrectnessHead.features(probs, dev_batch["marker_mask"], dev_batch["qtype"])
        confidence = torch.sigmoid(model.correctness(feats))

        for row, item in enumerate(chunk):
            width = len(item.markers)
            p = probs[row, :width].float().cpu().numpy()
            p = p / max(p.sum(), 1e-9)
            expected = float(sum(i * p[i] for i in range(width)))
            records.append(
                {
                    "case_id": item.case_id,
                    "workflow": item.workflow,
                    "qid": item.qid,
                    "qtype": item.qtype,
                    "probs": p.tolist(),
                    "target": list(item.target),
                    "label_index": item.label,
                    "confidence": float(confidence[row].cpu()),
                    "expected_level": expected,
                    "gold_score": item.gold_score,
                }
            )
    return records


def fit_temperature(model: VerdictModel, items: List[Item], pad_id: int, device: str, batch_size: int) -> None:
    """Grid-fit one temperature per (qtype, cardinality bucket) on the calibration fold."""
    model.eval()
    buckets: Dict[tuple, List[tuple]] = {}
    rng = random.Random(0)
    with torch.no_grad():
        for batch, chunk in batches(items, batch_size, pad_id, shuffle=False, rng=rng):
            dev_batch = to_device(batch, device)
            logits = model.option_logits(dev_batch).cpu()
            for row, item in enumerate(chunk):
                width = len(item.markers)
                key = (item.qtype, min(max(width, 2), 9) - 2)
                buckets.setdefault(key, []).append(
                    (logits[row, :width], torch.tensor(item.target, dtype=torch.float32))
                )

    grid = [round(0.25 + 0.05 * i, 2) for i in range(56)]  # 0.25 .. 3.00
    table = torch.ones(3, 8)
    for (qtype, bucket), rows in buckets.items():
        best_t, best_nll = 1.0, float("inf")
        for t in grid:
            nll = 0.0
            for logit, target in rows:
                nll += float(-(target * torch.log_softmax(logit / t, dim=-1)).sum())
            if nll < best_nll:
                best_nll, best_t = nll, t
        table[qtype, bucket] = best_t
        print(f"  temperature[qtype={qtype}, K bucket={bucket + 2}] = {best_t:.2f}  (n={len(rows)})")
    model.temperature.copy_(table.to(model.temperature.device))


def fit_correctness_head(model: VerdictModel, records: List[Dict], device: str, epochs: int = 300) -> None:
    """Train the correctness head on out-of-fold predictions from the calibration fold."""
    feats = []
    labels = []
    for r in records:
        p = torch.tensor(r["probs"], dtype=torch.float32).unsqueeze(0)
        mask = torch.ones_like(p, dtype=torch.bool)
        qtype = torch.tensor([r["qtype"]], dtype=torch.long)
        feats.append(CorrectnessHead.features(p, mask, qtype))
        labels.append(float(int(np.argmax(r["probs"])) == int(r["label_index"])))
    x = torch.cat(feats).to(device)
    y = torch.tensor(labels, dtype=torch.float32, device=device)

    head = model.correctness.to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=2e-3)
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(head(x), y)
        loss.backward()
        optimizer.step()
    print(f"  correctness head: final BCE {float(loss):.4f} over {len(labels)} calibration decisions")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Verdict 2.0.")
    parser.add_argument("--backbone", default="answerdotai/ModernBERT-base")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--accum", type=int, default=2)
    parser.add_argument("--lr_backbone", type=float, default=3e-5)
    parser.add_argument("--lr_head", type=float, default=1e-3)
    parser.add_argument("--lambda_brier", type=float, default=0.5)
    parser.add_argument("--lambda_hard", type=float, default=0.25)
    parser.add_argument("--lambda_rps", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=None, help="Cap training cases, for smoke tests.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--fp16", action="store_true", help="Mixed precision. Use on CUDA; Turing (GTX 16xx) has no bf16.")
    parser.add_argument("--optim8bit", action="store_true",
                        help="8-bit AdamW via bitsandbytes. Required to full fine-tune ModernBERT-large on a 6 GB card: "
                             "fp32 Adam states alone are 3.2 GB for a 396M model.")
    parser.add_argument("--grad_checkpoint", action="store_true", help="Trade compute for activation memory.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="artifacts/verdict2")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = pick_device(args.device)
    print(f"device={device} backbone={args.backbone}")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.backbone)
    pad_id = tokenizer.pad_token_id or 0

    print("Loading train split and partitioning by case id...")
    items = load_items("train", tokenizer, limit=args.limit)
    folds = split_by_case(items)
    print(f"  decisions: fit={len(folds['fit'])} calib={len(folds['calib'])} dev={len(folds['dev'])}")

    model = VerdictModel(args.backbone).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"  parameters: {params / 1e6:.1f}M")

    head_params = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    groups = [
        {"params": list(model.encoder.parameters()), "lr": args.lr_backbone},
        {"params": head_params, "lr": args.lr_head},
    ]
    if args.optim8bit:
        try:
            import bitsandbytes as bnb

            optimizer = bnb.optim.AdamW8bit(groups, weight_decay=0.01)
            print("  optimizer: AdamW8bit (bitsandbytes)")
        except ImportError as exc:
            raise SystemExit(
                "--optim8bit needs bitsandbytes: pip install bitsandbytes\n"
                f"  (import failed: {exc})"
            )
    else:
        optimizer = torch.optim.AdamW(groups, weight_decay=0.01)

    if args.grad_checkpoint:
        model.encoder.gradient_checkpointing_enable()
        print("  gradient checkpointing: on")
    steps = max(1, args.epochs * math.ceil(len(folds["fit"]) / (args.batch_size * args.accum)))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=[args.lr_backbone, args.lr_head], total_steps=steps, pct_start=0.1
    )
    use_amp = args.fp16 and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start = time.time()
    best = {"accuracy": -1.0}
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        running, seen, step = 0.0, 0, 0
        optimizer.zero_grad()
        for i, (batch, _) in enumerate(batches(folds["fit"], args.batch_size, pad_id, True, rng)):
            dev_batch = to_device(batch, device)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                logits = model.option_logits(dev_batch)
                loss = decision_loss(logits, dev_batch, args.lambda_brier, args.lambda_hard, args.lambda_rps)
            if not torch.isfinite(loss):
                raise ValueError("non-finite training loss")
            scaler.scale(loss / args.accum).backward()
            running += float(loss.detach())
            seen += 1
            if (i + 1) % args.accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if step < steps:
                    scheduler.step()
                step += 1

        dev_records = infer(model, folds["dev"], pad_id, device, args.batch_size, use_temperature=False)
        dev_metrics = summarize(dev_records)
        print(
            f"epoch {epoch + 1}/{args.epochs} loss {running / max(seen, 1):.4f} "
            f"dev_acc {dev_metrics['accuracy']:.4f} dev_brier {dev_metrics['brier']:.4f} "
            f"({time.time() - start:.0f}s)"
        )
        if dev_metrics["accuracy"] > best["accuracy"]:
            best = dev_metrics
            torch.save({"state_dict": model.state_dict(), "backbone": args.backbone, "args": vars(args)}, out_dir / "model.pt")

    print("\nRestoring best checkpoint and fitting calibration on the calib fold...")
    checkpoint = torch.load(out_dir / "model.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    fit_temperature(model, folds["calib"], pad_id, device, args.batch_size)
    calib_records = infer(model, folds["calib"], pad_id, device, args.batch_size, use_temperature=True)
    fit_correctness_head(model, calib_records, device)

    final = summarize(infer(model, folds["dev"], pad_id, device, args.batch_size, use_temperature=True))
    print("\nDEV metrics after calibration (test split untouched):")
    for key, value in final.items():
        print(f"  {key:<32} {value}")

    torch.save({"state_dict": model.state_dict(), "backbone": args.backbone, "args": vars(args)}, out_dir / "model.pt")
    tokenizer.save_pretrained(out_dir)
    (out_dir / "dev_metrics.json").write_text(json.dumps(final, indent=2))
    print(f"\nSaved to {out_dir}. Total wall time {time.time() - start:.0f}s.")


if __name__ == "__main__":
    main()
