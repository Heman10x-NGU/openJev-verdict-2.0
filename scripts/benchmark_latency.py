"""Experiment E7: Latency Benchmark with Rigorous Percentiles.

Benchmarks ONNX model inference latency across candidate menu cardinalities K in {3, 5, 9, 17, 25}.
Runs 20 warmup iterations and 200 timed iterations per setting.
Evaluates both FP32 and FP16 models under single-thread (WASM proxy) and multi-threaded CPU execution.
Reports p50, p90, p99, mean, stddev, system metadata, token counts, and ORT version.

Writes receipt to reports/v2/exp_e7_latency.json.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
from transformers import AutoTokenizer

from core.formatting import build_model_input

try:
    import onnxruntime as ort
    ORT_AVAILABLE = True
except ImportError:
    ORT_AVAILABLE = False


def run_latency_benchmark(
    checkpoint_dir: str = "artifacts/v2",
    reports_dir: str = "reports/v2",
    model_name: str = "knowledgator/gliclass-modern-base-v2.0",
    warmup_runs: int = 20,
    timed_runs: int = 200,
) -> dict[str, Any]:
    if not ORT_AVAILABLE:
        raise RuntimeError("onnxruntime is required for latency benchmarking.")

    rep_path = Path(reports_dir)
    rep_path.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    fp32_candidates = [
        Path(checkpoint_dir) / "model.onnx",
        Path(checkpoint_dir) / "openjev_modernbert.onnx",
    ]
    fp32_path = next((p for p in fp32_candidates if p.exists()), None)

    fp16_path = Path(checkpoint_dir) / "model_fp16.onnx"

    models_to_test = []
    if fp32_path and fp32_path.exists():
        models_to_test.append(("fp32", fp32_path))
    if fp16_path and fp16_path.exists():
        models_to_test.append(("fp16", fp16_path))

    if not models_to_test:
        raise FileNotFoundError(f"No ONNX models found in {checkpoint_dir}")

    cardinalities = [3, 5, 9, 17, 25]
    thread_configs = [1, 4]  # 1 thread proxies browser WASM; 4 threads proxies desktop

    benchmark_results: dict[str, Any] = {}
    sample_text = "I need to check why an international transfer of 500 EUR has not settled in the destination account."
    question = "Classify the following query into the appropriate category:"

    for model_precision, onnx_file in models_to_test:
        print(f"\nBenchmarking {model_precision.upper()} model ({onnx_file.name})...")
        model_results = {}

        for num_threads in thread_configs:
            thread_label = f"{num_threads}_thread"
            thread_results = {}
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = num_threads
            opts.inter_op_num_threads = 1
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            session = ort.InferenceSession(str(onnx_file), sess_options=opts, providers=["CPUExecutionProvider"])

            for k in cardinalities:
                labels = [f"candidate_category_description_{i}" for i in range(k - 1)] + ["insufficient evidence"]
                prompt = build_model_input(question, sample_text, labels)
                tokens = tokenizer([prompt], padding=True, truncation=True, return_tensors="np")
                input_ids = tokens["input_ids"]
                attention_mask = tokens["attention_mask"]
                token_count = int(attention_mask.sum())

                ort_inputs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                }

                # Warmup
                for _ in range(warmup_runs):
                    session.run(None, ort_inputs)

                # Timed runs
                durations_ms = []
                for _ in range(timed_runs):
                    t0 = time.perf_counter_ns()
                    session.run(None, ort_inputs)
                    t1 = time.perf_counter_ns()
                    durations_ms.append((t1 - t0) / 1e6)

                arr = np.array(durations_ms)
                stats = {
                    "k_cardinality": k,
                    "token_count": token_count,
                    "warmup_runs": warmup_runs,
                    "timed_runs": timed_runs,
                    "p50_ms": float(np.percentile(arr, 50)),
                    "p90_ms": float(np.percentile(arr, 90)),
                    "p99_ms": float(np.percentile(arr, 99)),
                    "mean_ms": float(np.mean(arr)),
                    "std_ms": float(np.std(arr)),
                    "min_ms": float(np.min(arr)),
                    "max_ms": float(np.max(arr)),
                }
                thread_results[f"k_{k}"] = stats
                print(
                    f"  [{thread_label}] K={k:2d} (tok={token_count:3d}) -> "
                    f"p50: {stats['p50_ms']:5.2f} ms | p90: {stats['p90_ms']:5.2f} ms | p99: {stats['p99_ms']:5.2f} ms | mean: {stats['mean_ms']:5.2f} ms"
                )

            model_results[thread_label] = thread_results

        benchmark_results[model_precision] = model_results

    receipt = {
        "experiment": "E7_latency_benchmark",
        "system_metadata": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "onnxruntime_version": ort.__version__,
        },
        "benchmarks": benchmark_results,
    }

    out_file = rep_path / "exp_e7_latency.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(receipt, f, indent=2)

    print(f"\nSaved E7 receipt to {out_file}")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description="Run E7 Latency Benchmark.")
    parser.add_argument("--checkpoint_dir", type=str, default="artifacts/v2")
    parser.add_argument("--reports_dir", type=str, default="reports/v2")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--runs", type=int, default=200)
    args = parser.parse_args()

    run_latency_benchmark(
        checkpoint_dir=args.checkpoint_dir,
        reports_dir=args.reports_dir,
        warmup_runs=args.warmup,
        timed_runs=args.runs,
    )


if __name__ == "__main__":
    main()
