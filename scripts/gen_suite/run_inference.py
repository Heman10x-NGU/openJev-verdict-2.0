"""Stage 2: Batched GPU inference and logit caching for Gen Suite.

Loads ONE model checkpoint (base or finetuned) exactly once, runs all 12 tasks
across all 3 framing arms (neutral, verbose, banking_framed), batches at size 32
sorted by token length to minimize padding, and writes raw logits to .npz caches.
"""

from __future__ import annotations

import argparse
import json
import time
import sys
from pathlib import Path
from typing import Any

WORKSPACE_DIR = Path(__file__).resolve().parent.parent.parent
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

from core.formatting import build_model_input
from gliclass import GLiClassModel
from scripts.gen_suite.tasks import TASK_REGISTRY, FramingArm, TaskItem, get_task_candidates

DATA_GEN_DIR = WORKSPACE_DIR / "data" / "gen"
CACHE_DIR = WORKSPACE_DIR / "reports" / "v2" / "_cache"

MODEL_BASE_NAME = "knowledgator/gliclass-modern-base-v2.0"
CHECKPOINT_PATHS = {
    "base": None,
    "finetuned": WORKSPACE_DIR / "artifacts" / "v2" / "model.safetensors",
}


def load_task_items(task_id: str, smoke: bool = False, max_items: int | None = None) -> list[TaskItem]:
    jsonl_path = DATA_GEN_DIR / f"{task_id}.jsonl"
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Dataset for {task_id} not found at {jsonl_path}. Run prepare_all.py first.")
    
    items = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(TaskItem.from_dict(json.loads(line)))
                
    if smoke:
        items = items[:20]
    elif max_items is not None:
        items = items[:max_items]
        
    return items


def run_task_arm_inference(
    model: GLiClassModel,
    tokenizer: Any,
    task_id: str,
    items: list[TaskItem],
    arm: FramingArm,
    device: torch.device,
    batch_size: int = 32,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Run inference for one task and one framing arm with length-sorted batching."""
    indexed_prompts = []
    for orig_idx, item in enumerate(items):
        if task_id == "T06":
            arm_candidates = item.candidates
        else:
            arm_candidates = get_task_candidates(task_id, arm)
        candidate_ids = [c["id"] for c in arm_candidates]
        candidate_labels = [c["description"] for c in arm_candidates]
        prompt = build_model_input(item.question, item.context, candidate_labels)
        target_idx = (
            candidate_ids.index(item.target_id)
            if item.target_id in candidate_ids
            else candidate_ids.index(INSUFFICIENT_EVIDENCE_ID)
        )
        indexed_prompts.append((orig_idx, prompt, item.id, target_idx, candidate_labels, candidate_ids))
        
    indexed_prompts.sort(key=lambda x: len(x[1]))
    
    ordered_logits = [None] * len(items)
    ordered_targets = [None] * len(items)
    ordered_ids = [None] * len(items)
    
    with torch.no_grad():
        for b_start in range(0, len(indexed_prompts), batch_size):
            b_chunk = indexed_prompts[b_start : b_start + batch_size]
            b_texts = [x[1] for x in b_chunk]
            
            tokens = tokenizer(
                b_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(device)
            
            outputs = model(**tokens)
            raw_logits = outputs.logits  # Shape: (batch_size, max_labels)
            
            for b_sub, (orig_idx, _, item_id, target_idx, candidate_labels, _) in enumerate(b_chunk):
                k_classes = len(candidate_labels)
                item_logits = raw_logits[b_sub, :k_classes].float().cpu().numpy()
                
                ordered_logits[orig_idx] = item_logits
                ordered_targets[orig_idx] = target_idx
                ordered_ids[orig_idx] = item_id
                
    max_k = max(l.shape[0] for l in ordered_logits)
    logits_arr = np.zeros((len(items), max_k), dtype=np.float32)
    for i, l in enumerate(ordered_logits):
        logits_arr[i, : l.shape[0]] = l
        
    targets_arr = np.array(ordered_targets, dtype=np.int64)
    default_cids = (
        [c["id"] for c in get_task_candidates(task_id, arm)]
        if task_id != "T06"
        else [c["id"] for c in items[0].candidates]
    )
    
    return logits_arr, targets_arr, ordered_ids, default_cids


def main():
    parser = argparse.ArgumentParser(description="Stage 2: Batched GPU inference and logit caching.")
    parser.add_argument("--checkpoint", choices=["base", "finetuned"], required=True, help="Checkpoint to evaluate.")
    parser.add_argument("--framing", choices=["all", "neutral", "verbose", "banking_framed"], default="all", help="Label framing arm to run.")
    parser.add_argument("--smoke", action="store_true", help="Run on smoke dataset (20 items per task).")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for model inference.")
    parser.add_argument("--device", type=str, default="mps", help="Device (mps, cuda, cpu).")
    args = parser.parse_args()
    
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    # Device selection
    if args.device == "mps" and not torch.backends.mps.is_available():
        device = torch.device("cpu")
    elif args.device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
        
    print(f"=== Stage 2: Running Inference for Checkpoint '{args.checkpoint}' on {device.type.upper()} ===")
    
    # 1. Load model once
    t0_load = time.perf_counter()
    print(f"Loading base architecture from '{MODEL_BASE_NAME}'...")
    model = GLiClassModel.from_pretrained(MODEL_BASE_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_BASE_NAME)
    
    if args.checkpoint == "finetuned":
        ckpt_path = CHECKPOINT_PATHS["finetuned"]
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Fine-tuned checkpoint not found at {ckpt_path}")
        print(f"Loading fine-tuned safetensors from {ckpt_path}...")
        state_dict = load_file(str(ckpt_path))
        model.load_state_dict(state_dict)
        
    model.to(device)
    model.eval()
    load_time = time.perf_counter() - t0_load
    print(f"Model loaded and moved to {device.type.upper()} in {load_time:.2f}s.\n")
    
    arms_to_run: list[FramingArm] = (
        ["neutral", "verbose", "banking_framed"]
        if args.framing == "all"
        else [args.framing]  # type: ignore
    )
    
    total_cached_rows = 0
    t0_infer = time.perf_counter()
    
    for task_id in TASK_REGISTRY.keys():
        items = load_task_items(task_id, smoke=args.smoke)
        for arm in arms_to_run:
            t0_task = time.perf_counter()
            logits, targets, item_ids, candidate_ids = run_task_arm_inference(
                model=model,
                tokenizer=tokenizer,
                task_id=task_id,
                items=items,
                arm=arm,
                device=device,
                batch_size=args.batch_size,
            )
            elapsed_task = time.perf_counter() - t0_task
            throughput = len(items) / elapsed_task if elapsed_task > 0 else 0
            
            cache_file = CACHE_DIR / f"logits_{args.checkpoint}_{arm}_{task_id}.npz"
            np.savez_compressed(
                cache_file,
                logits=logits,
                target_idx=targets,
                item_ids=np.array(item_ids),
                candidate_ids=np.array(candidate_ids),
                task_id=task_id,
                checkpoint=args.checkpoint,
                arm=arm,
            )
            total_cached_rows += len(items)
            print(f"[{args.checkpoint}][{arm:<14}][{task_id}] -> Saved {cache_file.name} | Shape: {logits.shape} | {throughput:.1f} items/s")
            
    total_infer_time = time.perf_counter() - t0_infer
    print(f"\nCompleted {args.checkpoint} sweep in {total_infer_time:.2f}s wall-clock ({total_cached_rows} total logit rows cached).")


if __name__ == "__main__":
    main()
