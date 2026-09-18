#!/usr/bin/env python3
"""Run baseline evaluation 2 more times (Run 2 and Run 3) and calculate aggregate mean and std."""

import json
import subprocess
import sys
import time
from pathlib import Path
import numpy as np

def run_pass(run_num: int) -> dict:
    out_file = f"reports/verdict_baseline_run_{run_num}.json"
    cmd = [
        ".venv/bin/python",
        "scripts/evaluate_verdict_baseline.py",
        "--out", out_file
    ]
    print(f"\n==========================================")
    print(f"       STARTING BASELINE RUN {run_num}")
    print(f"==========================================")
    t0 = time.perf_counter()
    res = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.perf_counter() - t0
    if res.returncode != 0:
        print(f"Run {run_num} failed:\n{res.stderr}", file=sys.stderr)
        raise RuntimeError(f"Run {run_num} failed")
    
    print(res.stdout)
    with open(out_file, "r") as f:
        data = json.load(f)
    data["wall_clock_time"] = elapsed
    return data

def main():
    # Load Run 1
    run1_path = Path("reports/verdict_baseline_benchmark_full.json")
    if not run1_path.exists():
        print("Run 1 not found, running Run 1 first...")
        run1 = run_pass(1)
    else:
        with open(run1_path) as f:
            run1 = json.load(f)
        print("Loaded Run 1 metrics successfully.")

    # Brief cooldown between runs to prevent thermal throttling
    time.sleep(3)
    run2 = run_pass(2)
    time.sleep(3)
    run3 = run_pass(3)

    all_runs = [run1, run2, run3]
    accs = [r["metrics"]["accuracy"] for r in all_runs]
    soft_accs = [r["metrics"]["soft_accuracy"] for r in all_runs]
    briers = [r["metrics"]["brier_score"] for r in all_runs]
    eces = [r["metrics"]["ece"] for r in all_runs]
    lat_p50s = [r["metrics"]["latency_p50_ms"] for r in all_runs]
    lat_p95s = [r["metrics"]["latency_p95_ms"] for r in all_runs]

    summary = {
        "num_runs": 3,
        "runs": all_runs,
        "summary": {
            "accuracy": {"mean": float(np.mean(accs)), "std": float(np.std(accs))},
            "soft_accuracy": {"mean": float(np.mean(soft_accs)), "std": float(np.std(soft_accs))},
            "brier_score": {"mean": float(np.mean(briers)), "std": float(np.std(briers))},
            "ece": {"mean": float(np.mean(eces)), "std": float(np.std(eces))},
            "latency_p50_ms": {"mean": float(np.mean(lat_p50s)), "std": float(np.std(lat_p50s))},
            "latency_p95_ms": {"mean": float(np.mean(lat_p95s)), "std": float(np.std(lat_p95s))}
        }
    }

    summary_file = Path("reports/verdict_baseline_3runs_summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("       3-RUN VERDICT 1.0 FINAL BASELINE SUMMARY")
    print("=" * 60)
    print(f"Top-1 Accuracy    : {summary['summary']['accuracy']['mean']*100:.2f}% ± {summary['summary']['accuracy']['std']*100:.2f}%")
    print(f"Soft Accuracy     : {summary['summary']['soft_accuracy']['mean']*100:.2f}% ± {summary['summary']['soft_accuracy']['std']*100:.2f}%")
    print(f"Brier Score       : {summary['summary']['brier_score']['mean']:.4f} ± {summary['summary']['brier_score']['std']:.4f}")
    print(f"ECE               : {summary['summary']['ece']['mean']:.4f} ± {summary['summary']['ece']['std']:.4f}")
    print(f"Latency P50       : {summary['summary']['latency_p50_ms']['mean']:.1f} ± {summary['summary']['latency_p50_ms']['std']:.1f} ms")
    print("=" * 60)
    print(f"Summary written to {summary_file}")

if __name__ == "__main__":
    main()
