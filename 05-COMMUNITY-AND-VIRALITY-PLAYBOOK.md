# Community and virality playbook

This document provides the release strategy, communication guidelines, and interactive demo architecture to build technical credibility and engagement on X.

## The current landscape on X

The public reception of TypeSafe AI's Jev and subsequent community reactions created a unique environment:
* TypeSafe AI generated massive curiosity by promising 200x cheaper, 100x faster System 1 decisions, but locked access behind a private enterprise waitlist.
* Harsha Gundala announced a two-hour open-source alternative, gained quick virality, and was subsequently scrutinized by senior engineers when source inspection revealed empty model repositories, faked 75% calibration fallbacks, and MLX key-value cache buffer bugs.
* Builders on X repeatedly asked for two things: honest probabilistic calibration and an interactive browser-native demo running on client hardware.

Shipping an open, working solution that directly answers these community demands will attract genuine technical attention.

---

## The four pillars of technical credibility

To avoid the skepticism that faced earlier attempts, follow these standards:

### 1. Upload verified Safetensors weights
Do not upload empty repositories with inference wrappers. Upload full, trained model weights to Hugging Face:
* Formats: Standard `model.safetensors`, GGUF (for llama.cpp / Ollama), and ONNX (for WebGPU).
* Documentation: Provide a complete Hugging Face model card with parameter counts, architecture details, and training data provenance.

### 2. Publish empirical calibration benchmarks
Publish genuine calibration statistics rather than asserting that the model cannot fail:
* Report Expected Calibration Error (ECE) across ten confidence bins on a public test dataset (such as Banking77 or synthetic triage benchmarks).
* Include reliability diagrams showing how predicted confidence correlates with empirical accuracy.
* Transparently report out-of-distribution performance when inputs match none of the allowed options.

### 3. Provide an interactive WebGPU demo
The strongest driver of organic reach on developer social media is zero-friction interaction:
* Convert the 139M ModernBERT model to ONNX format using `optimum-cli`.
* Build a static HTML and JavaScript interface using `transformers.js` running on WebGPU.
* Host the demo on GitHub Pages.
* Visitors paste arbitrary text, specify candidate categories, and receive typed classifications with animated probability bars in under 15 milliseconds on their local GPU.
* The demo operates entirely in the user's browser with zero API keys, no login walls, and zero server infrastructure cost.

### 4. Provide a single-file reproducible training script
Publish a clean Python script (`train_rlcd.py`) that anyone can execute on a local Mac or a free Kaggle notebook:
* Downloads base weights from Hugging Face.
* Loads the synthetic calibration dataset.
* Trains the model using composite Brier score loss.
* Runs post-training temperature scaling.
* Outputs the final calibrated checkpoint in under 60 minutes.

---

## Structure of the release thread on X

Organize your announcement thread to address technical questions upfront:

1. The hook: Announce an open-source, 139M parameter System 1 decision engine trained with genuine Reinforcement Learning for Calibrated Decisions, running in under 15 milliseconds in the browser.
2. The problem: Explain why forcing conversational LLMs to generate strings for structured decisions wastes tokens and creates parsing errors.
3. The technical solution: Explain the bidirectional ModernBERT encoder, the composite Brier score loss function, and why non-autoregressive joint encoding eliminates JSON syntax failures.
4. The live demo: Link directly to the GitHub Pages WebGPU demo so readers can test their own inputs immediately.
5. The weights and code: Link to the Hugging Face model repository and GitHub code repository.
6. The benchmark receipt: Include the ECE calibration curve, demonstrating that 80% predicted confidence corresponds to 80% empirical accuracy.
