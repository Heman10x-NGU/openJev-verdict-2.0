# openJev-verdict-2.0: Non-Autoregressive System 1 Decision Engine

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-heman10x%2FopenJev--verdict--2.0-blue)](https://huggingface.co/heman10x/openJev-verdict-2.0)
[![GitHub Repository](https://img.shields.io/badge/GitHub-openJev--verdict--2.0-black?logo=github)](https://github.com/Heman10x-NGU/openJev-verdict-2.0)
[![Accuracy](https://img.shields.io/badge/Top--1%20Accuracy-77.10%25%20(SOTA)-brightgreen)](#the-openjev-verdict-20-benchmark-breakthrough)
[![Calibration](https://img.shields.io/badge/Confidence%20ECE-1.44%25%20(15x%20SOTA)-success)](#dual-channel-calibration)
[![Latency](https://img.shields.io/badge/Latency-~20ms%20%2F%20decision-orange)](#single-pass-efficiency)
[![In-Browser WebGPU](https://img.shields.io/badge/WebGPU-Zero--Cloud%20Edge%20Ready-blueviolet)](#in-browser-webgpu-engine)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue)](LICENSE)

**openJev-verdict-2.0** is an open-source, non-autoregressive foundational decision model engineered for deterministic software automation, inspired by **TypeSafe AI's Jev** and **Reinforcement Learning for Calibrated Decisions (RLCD)**.

While standard Large Language Models waste compute generating text strings that deterministic software must parse and validate, openJev-verdict-2.0 evaluates typed decision schemas (`Choice`, `Score`, `Noul`) in **non-autoregressive forward passes (~20–25 ms per decision)** with **actuarial-grade calibration (1.44% ECE)**.

<p align="center">
  <img src="assets/verdict2_vs_laya_jev_showdown.png" alt="openJev-verdict-2.0 vs Laya and Jev Showdown" width="880">
</p>

---

## The openJev-verdict-2.0 Benchmark Breakthrough

Evaluated on the held-out test split of `LocalLLaMA/typed-decisions` (N = 2,000 decisions across enterprise financial, security, customer support, and agent trace observability workflows):

| Model | Parameters | Top-1 Accuracy ↑ | Brier Loss (Soft) ↓ | ECE (Correctness Head) ↓ | ECE (Distribution Channel) ↓ | Latency (Decision) ↓ | Option Flip Rate ↓ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Verdict 1.0 Baseline** | 151M | 26.10% | 0.5851 | — | 0.4209 | ~35 ms | High |
| **TF-IDF + Logistic Reg** | — | 66.10% | 0.1520 | — | 0.0207 | **~8 ms** | **0.00%** |
| **TypeSafe Jev 1.13.0** <sup>†</sup> | ~150M | 72.70% | 0.1480 | — | 0.1440 | ~140 ms | — |
| **Laya (ModernBERT-large)**| 421.3M | 76.60% | 0.0660 | — | 0.2140 | ~31 ms | — |
| **openJev-verdict-2.0 (Ours)**| **149.6M** | **77.10%** | **0.0636** | **0.0144** | **0.1513** | **~20–25 ms** | **4.76%** |

*Notes:*  
<sup>†</sup> *Vendor baseline cataloged in Laya's published evaluation suite.*  
*Kev published a 7.41% flip rate on its multi-task suite ($N=81$); openJev-verdict-2.0 achieves 4.76% across $N=2,918$ enterprise perturbations.*

<p align="center">
  <img src="assets/benchmark_breakthrough_twitter.png" alt="openJev-verdict-2.0 Highlights" width="880">
</p>

---

## The Five Core Architectural Breakthroughs

### 1. 2.8x Parameter Efficiency (Base Beats Large)
- Laya reached 76.60% accuracy using `ModernBERT-large` (421.3M parameters).
- **openJev-verdict-2.0 achieves 77.10% accuracy and 0.0636 Brier score** using `ModernBERT-base` (**149.6M parameters**) — matching and outperforming a model with 2.8x more parameters using less than half the VRAM.
- Fully trained in 8.8 hours on a budget consumer laptop GPU (GTX 1660 Ti, 6GB VRAM) with zero cloud clusters.

### 2. Dual-Channel Calibration: Solving the Soft-Label Trap
Soft-labeled benchmarks create an inherent mathematical dilemma:
- Human expert panels have measured doubt (average top probability is 0.659).
- If a model matches the panel distribution, its top probability sits near ~0.62 while its empirical accuracy is 77.10%. Evaluated against binary correctness, its calibration error explodes (**Laya's trap: Brier 0.0660, ECE 0.2140**).
- If a model sharpens its probabilities to match accuracy, it diverges from the panel and degrades Brier loss (**Jev's trap: ECE 0.1440, Brier 0.1480**).
- **openJev-verdict-2.0 solves this by decoupling into dual channels**:
  - **Channel 1 (Distribution Head)**: Marker-pointer logits scoring **0.0636 Brier** (and 0.1513 distribution ECE, beating Laya's 0.2140 by 29%).
  - **Channel 2 (Correctness Head)**: Dedicated MLP trained out-of-fold over prediction geometry (entropy, margin, cardinality), achieving **1.44% ECE (0.0144)** and **0.7861 AUROC**.

### 3. Selective Classification & Automated Gating
The confidence head enables reliable human-in-the-loop escalation policies in deterministic software:
- **At 80% coverage**: Model accuracy rises to **85.00%**.
- **At 60% coverage**: Model accuracy reaches **90.21%**.
Deterministic code can automate high-confidence decisions and escalate low-confidence uncertainty to human operators without false-positive surprises.

### 4. Symmetric Permutation-KL (Crushing Prompt-Order Bias)
Autoregressive decoders suffer from order bias: shuffling option order (A/B/C vs C/B/A) flips decisions frequently (Kev-0.5B: 7.41% flip rate).
- openJev-verdict-2.0 applies symmetric KL-divergence penalties between twin option-shuffled batches during training.
- Evaluated on 2,918 perturbed test decisions, the argmax flip rate dropped to **4.76%** (a 36% reduction), with a 90th-percentile probability spread of **0.0915** (vs Kev's 0.2486).

### 5. In-Browser WebGPU Execution (<600 MB)
Because it is an efficient 149.6M non-autoregressive encoder rather than an autoregressive LLM, openJev-verdict-2.0 runs locally in client browsers via WebGPU:
- **Zero cloud API costs**: No tokens to pay for.
- **Zero data exfiltration**: Private invoices, security logs, and PII never leave the client's machine.
- **Sub-35 ms execution**: Real-time evaluation in Chrome/Edge.

---

## Interactive Dashboard

Explore the interactive Pareto frontier scatter plot, KPI cards, and full metric receipts in the self-contained dashboard:
- [Interactive Benchmark Dashboard](docs/index.html) (or via GitHub Pages at `https://heman10x-ngu.github.io/openJev-verdict-2.0/`)

---

## Quickstart

### Python Usage

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("heman10x/openJev-verdict-2.0")

# Input sequence layout:
# [CLS] <type> <question> [SEP] [MASK]<opt0> [MASK]<opt1> ... [SEP] <state> [SEP]
```

### In-Browser WebGPU Playground

1. Navigate to `webgpu-demo/`:
```bash
cd webgpu-demo
python3 -m http.server 8080
```
2. Open `http://localhost:8080` in Chrome or any WebGPU-enabled browser to test live customer service, security, and invoice decisions directly on your device.

---

## Citation & Upstream Credits

- Inspired by **TypeSafe AI's Jev** architecture and **RLCD (Reinforcement Learning for Calibrated Decisions)**.
- Base backbone: ModernBERT (`answerdotai/ModernBERT-base`).
- Benchmark dataset: `LocalLLaMA/typed-decisions`.
- License: Apache 2.0.
