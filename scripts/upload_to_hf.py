#!/usr/bin/env python3
import json
import sys
from pathlib import Path
from huggingface_hub import HfApi

def main():
    api = HfApi()
    repo_id = "heman10x/rlcd-modernbert-151m"

    with open("artifacts/ARTIFACTS.json", "r", encoding="utf-8") as f:
        manifest = json.load(f)

    for target_name, meta in manifest["files"].items():
        src_file = meta["source_file"]
        print(f"Uploading {src_file} -> {target_name}...")
        api.upload_file(
            path_or_fileobj=src_file,
            path_in_repo=target_name,
            repo_id=repo_id,
            repo_type="model"
        )
        print(f"Uploaded {target_name}")

    model_card = """---
license: apache-2.0
base_model: knowledgator/gliclass-modern-base-v2.0
library_name: transformers
pipeline_tag: text-classification
tags:
- rlcd
- typesafe-ai
- jev
- decision-engine
- system-1
- modernbert
- gliclass
- non-autoregressive
- zero-token-generation
- structured-outputs
- calibration
- expected-calibration-error
- ece
- brier-score
- proper-scoring-rules
- onnx
- webgpu
- edge-ai
- fast-inference
- banking77
---

# OpenJev (Verdict): Non-Autoregressive Decision Engine (151M)

[![GitHub Repository](https://img.shields.io/badge/GitHub-Verdict--open--jev-black?logo=github)](https://github.com/Heman10x-NGU/Verdict-open-jev)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-heman10x%2Frlcd--modernbert--151m-blue)](https://huggingface.co/heman10x/rlcd-modernbert-151m)
[![WebGPU Demo](https://img.shields.io/badge/WebGPU-In--Browser%20Playground-green)](https://github.com/Heman10x-NGU/Verdict-open-jev#running-the-in-browser-webgpu-playground)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue)](https://github.com/Heman10x-NGU/Verdict-open-jev/blob/main/LICENSE)

**OpenJev (Verdict)** is an open-source, post-trained foundational decision model built for structured software workflows, inspired by **TypeSafe AI's Jev** and **Reinforcement Learning for Calibrated Decisions (RLCD)**. It provides calibrated semantic judgments (discrete choices, ordinal scores, and binary probabilities) in a single forward pass without conversational text generation.


- **GitHub Repository**: [https://github.com/Heman10x-NGU/Verdict-open-jev](https://github.com/Heman10x-NGU/Verdict-open-jev)
- **Base Architecture**: ModernBERT-base (`knowledgator/gliclass-modern-base-v2.0`, 151,378,177 parameters)
- **Logit Capacity**: 25 candidate slots (24 substantive options + 1 explicit abstention slot)

---

## Key Features

1. **Non-Autoregressive Single Pass**: Evaluates all candidate options simultaneously in a single forward pass (< 35ms latency) without token generation loops.
2. **Proper Scoring Calibration**: Trained with composite Cross-Entropy + Brier Score loss:
   $$\\mathcal{L}_{\\text{total}} = \\mathcal{L}_{\\text{CE}} + 1.0 \\times \\mathcal{L}_{\\text{Brier}}$$
   followed by post-hoc L-BFGS temperature scaling ($T = 1.0716$).
3. **Explicit Abstention Route**: Dedicated `__insufficient_evidence__` candidate slot ensures calibrated rejection on out-of-distribution or insufficient context queries.
4. **Edge and In-Browser WebGPU**: Runs locally in browsers via WebGPU/WASM and on servers via PyTorch/ONNX Runtime.

---

## Quickstart

### Python SDK

```bash
git clone https://github.com/Heman10x-NGU/Verdict-open-jev.git
cd Verdict-open-jev
pip install -e .
python scripts/download_artifacts.py
```

```python
from rlcd import DecisionEngine, Choice, Option

engine = DecisionEngine()
query = Choice(
    question="What is the primary customer inquiry?",
    options=[
        Option(id="card_lost", description="Reporting a lost or stolen card"),
        Option(id="dispute_charge", description="Disputing an unrecognized charge"),
        Option(id="pin_reset", description="Requesting a PIN reminder or reset"),
    ]
)
result = engine.evaluate(
    context="I lost my wallet yesterday and need to stop my debit card immediately.",
    queries=[query]
)

print(f"Selected: {result.results[0].selected_option_id}")
print(f"Confidence: {result.results[0].confidence:.4f}")
print(f"Abstention probability: {result.results[0].p_abstain:.4f}")
```

### Direct ONNX Runtime Loading

```python
import onnxruntime as ort
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(repo_id="heman10x/rlcd-modernbert-151m", filename="model.onnx")
session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
```

---

## Empirical Benchmark Results

All metrics reflect evaluation on the held-out test split (1,000 cases, 5 candidates) and out-of-scope challenge sets:

| Metric | Uncalibrated | Calibrated | 95% Bootstrap CI |
| :--- | :--- | :--- | :--- |
| **Top-1 Accuracy** | 95.00% | 95.00% | [93.60%, 96.20%] |
| **Negative Log-Likelihood (NLL)** | 0.1787 | 0.1768 | [0.1345, 0.2223] |
| **Multi-Class Brier Score** | 0.0790 | 0.0785 | [0.0601, 0.0978] |
| **Equal-Width ECE (10 bins)** | 3.52% | 3.35% | [2.58%, 4.56%] |
| **Adaptive ECE (10 bins)** | 3.50% | 3.32% | [2.55%, 4.49%] |
| **Out-of-Scope Abstention Recall** | 97.50% | 97.50% | [95.07%, 99.49%] |
| **Out-of-Scope Abstention Precision** | 89.45% | 89.45% | [85.33%, 93.36%] |
| **Inference Latency (p50)** | 35.58 ms | 35.58 ms | Single-pass forward |
| **Inference Latency (p95)** | 39.81 ms | 39.81 ms | Single-pass forward |

---

## Citation and Upstream Credits

- Inspired by **TypeSafe AI's Jev** architecture and **RLCD (Reinforcement Learning for Calibrated Decisions)**.
- Base encoder backbone: ModernBERT (`knowledgator/gliclass-modern-base-v2.0`).
- Evaluation benchmarks: PolyAI Banking77 and CLINC150 Out-of-Scope datasets.
"""

    api.upload_file(
        path_or_fileobj=model_card.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model"
    )
    print("Uploaded README.md (Model card)")

if __name__ == "__main__":
    main()

