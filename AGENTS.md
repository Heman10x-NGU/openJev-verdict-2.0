# openJev-verdict-2.0: Developer & Agent Context

This repository houses **openJev-verdict-2.0**, an open-source, non-autoregressive foundational decision model engineered for deterministic software workflows. It is inspired by TypeSafe AI's Jev and Reinforcement Learning for Calibrated Decisions (RLCD).

## Project Overview

Generative Large Language Models spend excessive compute producing unstructured text that deterministic code must subsequently parse, validate, and retry. 

openJev-verdict-2.0 eliminates open-ended string generation. The engine accepts an input context and multiple typed question schemas (`Choice`, `Score`, `Noul`), returning discrete classifications, ordinal ranks, and binary decisions with actuarial-grade confidence calibration in non-autoregressive forward passes.

## Benchmark & Performance Architecture

* **Backbone:** ModernBERT-base (`answerdotai/ModernBERT-base`, 149.6M parameters).
* **Benchmark:** `LocalLLaMA/typed-decisions` (2,000 held-out test decisions across enterprise financial, customer service, security, and agent trace observability workflows).
* **Hardware Footprint:** Fine-tuned in 8.8 hours on a single consumer laptop GPU (GTX 1660 Ti, 6 GB VRAM) with zero cloud clusters.
* **Top-1 Accuracy:** 77.10% (matches and outperforms 421M Laya at 76.60% and Jev at 72.70%).
* **Dual-Channel Calibration:**
  * **Channel 1 (Distribution):** Brier score 0.0636, Distribution ECE 15.13% (29% lower error than Laya's 21.40%).
  * **Channel 2 (Confidence Head):** 1.44% ECE dedicated correctness gate for automated routing.
* **Order Stability:** Argmax flip rate of 4.76% under option permutations (36% reduction compared to Kev-0.5B).

## Core Invariants

1. **Non-Autoregressive Inference:** All target questions are evaluated without sequential token generation loops.
2. **Proper Scoring Rules:** Calibration values represent empirical probabilities under strictly proper scoring rules. Confidence values are never clamped or fabricated.
3. **Explicit Abstention:** Schemas provide an explicit route for insufficient evidence to maintain calibrated probabilities on out-of-distribution inputs.
4. **Deterministic Policy Control:** The model supplies bounded semantic judgment; deterministic software maintains state, business rules, arithmetic, and execution side-effects.

## Repository Layout

* `docs/`: Self-contained interactive benchmark dashboard and visualization assets.
* `verdict2/`: Model architecture (`model.py`), sequence builders (`data.py`), loss functions (`losses.py`), and evaluation engine (`evaluate.py`).
* `artifacts/verdict2-base/`: Model weights, tokenizer configs, and evaluation metrics.
* `reports/`: Verifiable JSON benchmark receipts and out-of-fold sweep logs.
* `scripts/`: Data preparation, training, calibration, and selective-classification evaluation scripts.
