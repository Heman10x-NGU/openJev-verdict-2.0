# Walkthrough: Real benchmark accuracy, calibration receipts, and browser demo

This document records the empirical fine-tuning, calibration, and ONNX WebGPU deployment for **Verdict-open-jev** (ModernBERT-base + GLiClass-v2, 151M parameters) evaluated on PolyAI **Banking77** and **CLINC150** out-of-scope benchmarks.

---

## 1. Empirical benchmark receipts

Evaluated on **1,000 held-out real customer queries** (`data/real_banking_test.jsonl`), consisting of 800 banking intent queries across 77 categories and 200 out-of-scope queries mapped to explicit abstention (`__insufficient_evidence__`).

Model checkpoint selection was conducted under proper scoring rules (validation negative log-likelihood, selecting Epoch 2).

<!-- BEGIN GENERATED: walkthrough_benchmark_receipts -->
| Metric | Uncalibrated | Calibrated | 95% Bootstrap CI | Notes / Definition |
| :--- | :--- | :--- | :--- | :--- |
| Top-1 accuracy | 95.00% | 95.00% | [93.60%, 96.20%] | Evaluated across 1,000 held-out test examples |
| Negative log-likelihood (NLL) | 0.1731 | 0.1768 | N/A | Strictly proper scoring rule across candidates |
| Multiclass Brier score | 0.0756 | 0.0785 | N/A | Mean squared probability error |
| Equal-width ECE (10 bins) | 1.13% | 3.35% | [2.58%, 4.56%] | Standard 10-bin expected calibration error |
| Equal-mass ECE (10 bins) | 0.83% | 2.87% | [2.15%, 4.01%] | Quantile-binned calibration error |
| Adaptive ECE (min 10 samples) | 0.83% | 2.87% | N/A | Excludes bins with fewer than 10 samples |
| Maximum Calibration Error (MCE) | 15.53% | 15.45% | N/A | Bins with >= 10 samples only |
| Out-of-scope abstention recall | 97.50% | 97.50% | [95.07%, 99.49%] | Proportion of out-of-scope queries flagged |
| Out-of-scope abstention precision | 89.45% | 89.45% | [85.33%, 93.36%] | Proportion of abstention predictions that are correct |
| Out-of-scope abstention F1 | 93.30% | 93.30% | N/A | Harmonic mean of abstention recall and precision |
| False abstentions (in-scope) | 23 | 23 | N/A | In-scope banking queries erroneously flagged to abstain |
| Optimal temperature ($T$) | 1.0000 | 1.4265 | N/A | Fitted on calibration set NLL via L-BFGS |
<!-- END GENERATED: walkthrough_benchmark_receipts -->

### Challenge slices evaluation

Candidate menu scaling and out-of-scope robustness across candidate cardinalities K in {3, 5, 9, 17, 25}:

<!-- BEGIN GENERATED: walkthrough_slices -->
| Candidate menu size | Top-1 accuracy | Brier score | Equal-width ECE | False abstentions | Sample count |
| :--- | :--- | :--- | :--- | :--- | :--- |
| K = 3 | 97.00% | 0.0478 | 5.92% | 1 | 100 |
| K = 5 | 96.00% | 0.0829 | 3.61% | 1 | 100 |
| K = 9 | 91.00% | 0.1378 | 6.13% | 1 | 100 |
| K = 17 | 78.00% | 0.3004 | 12.58% | 4 | 100 |
| K = 25 (maximum capacity) | 72.00% | 0.3618 | 7.84% | 4 | 100 |
<!-- END GENERATED: walkthrough_slices -->

---

## 2. Empirical robustness and sensitivity experiments

The following table records the empirical evidence addressing the core open questions:

<!-- BEGIN GENERATED: walkthrough_experiments -->
| Experiment / Question | Baseline / Gold | Measured Result | Boundary / Finding | Source Receipt |
| :--- | :--- | :--- | :--- | :--- |
| **E1**: Shuffled-context control | Original Acc: 95.00% | Shuffled Acc: 22.40% | Accuracy drops by 72.60 points; model abstains 72.70% | `reports/v2/exp_e1_shuffled_control.json` |
| **E2**: Option-order sensitivity | Reversal flips: 3.00% | Any permutation: 4.50% | Mean TV distance: 0.0337; flips concentrate at low confidence (58.9%) | `reports/v2/exp_e2_option_order.json` |
| **E3**: Label-length & glossary bias | Curated Acc: 94.31% | Templated Acc: 96.79% | Length correlation Pearson r=0.1027; uniform glossary rerun Acc: 93.00% | `reports/v2/exp_e3_label_bias.json` |
| **E4**: Hard-negative distractors | Random K=5 Acc: 96.00% | Hard K=5: 94.75% / Hard K=9: 84.75% | Confusable non-gold distractors cause a 10.00 point accuracy drop from K=5 to K=9 | `reports/v2/exp_e4_hard_negatives.json` |
| **E5**: Abstention generalization | Baseline Recall: 75.50% | Hard K=9: 18.00% / Synonym: 23.50% | Combined recall collapses to 10.00%; abstention does not generalize across synonyms | `reports/v2/exp_e5_abstention_generalization.json` |
| **E6**: Contamination audit | Exact matches: 0 / 2.3M | Jaccard >= 0.8: 3 pairs | Decontaminated test accuracy moves by +0.09 points (95.09%) | `reports/v2/exp_e6_contamination.json` |
| **E7**: Latency percentiles (K=5) | Prior claim: Unverified point estimate | Single-thread p50: 35.58 ms | Rigorous multi-trial distribution; WASM single-thread proxy measured across K in {3,5,9,17,25} | `reports/v2/exp_e7_latency.json` |
| **E8**: FP16 quality parity | FP32 test Acc: 94.50% | FP16 test Acc: 94.50% | Max delta across all splits is 0.00%; recommended: SHIP_FP16 | `reports/v2/exp_e8_fp16_parity.json` |
| **E9**: External TypeSafe evals | TypeSafe Jev: 90.80% | Verdict-open-jev: 48.07% | 151M encoder zero-shot floor measured against 26B DiffusionGemma (88.43%) across 337 cases | `reports/v2/exp_e9_external_cases.json` |
<!-- END GENERATED: walkthrough_experiments -->

---

## 3. Measured inference latency distribution

Latency benchmarks evaluated across 20 warmup runs and 200 timed runs per setting on Apple Silicon hardware via ONNX Runtime:

<!-- BEGIN GENERATED: walkthrough_latency -->
| Model format | Execution threads | Menu size | p50 latency | p90 latency | p99 latency | Mean latency | Sample tokens |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| FP32 | 1 thread (WASM proxy) | K = 3 | 27.58 ms | 27.82 ms | 30.70 ms | 27.77 ms | 62 tokens |
| FP32 | 1 thread (WASM proxy) | K = 5 | 35.58 ms | 36.09 ms | 42.72 ms | 35.91 ms | 80 tokens |
| FP32 | 1 thread (WASM proxy) | K = 9 | 49.04 ms | 58.08 ms | 62.12 ms | 51.87 ms | 116 tokens |
| FP32 | 1 thread (WASM proxy) | K = 17 | 92.16 ms | 93.13 ms | 108.19 ms | 92.33 ms | 188 tokens |
| FP32 | 1 thread (WASM proxy) | K = 25 | 140.10 ms | 148.47 ms | 161.13 ms | 141.76 ms | 260 tokens |
| FP32 | 4 thread (multi-thread) | K = 3 | 29.98 ms | 34.63 ms | 35.85 ms | 30.93 ms | 62 tokens |
| FP32 | 4 thread (multi-thread) | K = 5 | 49.47 ms | 71.17 ms | 170.35 ms | 54.36 ms | 80 tokens |
| FP32 | 4 thread (multi-thread) | K = 9 | 63.27 ms | 102.21 ms | 166.83 ms | 72.42 ms | 116 tokens |
| FP32 | 4 thread (multi-thread) | K = 17 | 132.10 ms | 162.68 ms | 221.32 ms | 130.88 ms | 188 tokens |
| FP32 | 4 thread (multi-thread) | K = 25 | 129.93 ms | 141.04 ms | 182.99 ms | 133.61 ms | 260 tokens |
| FP16 | 1 thread (WASM proxy) | K = 3 | 77.56 ms | 81.81 ms | 92.04 ms | 77.79 ms | 62 tokens |
| FP16 | 1 thread (WASM proxy) | K = 5 | 99.77 ms | 101.38 ms | 117.84 ms | 100.13 ms | 80 tokens |
| FP16 | 1 thread (WASM proxy) | K = 9 | 146.50 ms | 151.94 ms | 166.63 ms | 147.34 ms | 116 tokens |
| FP16 | 1 thread (WASM proxy) | K = 17 | 251.53 ms | 267.43 ms | 293.57 ms | 254.87 ms | 188 tokens |
| FP16 | 1 thread (WASM proxy) | K = 25 | 425.41 ms | 605.09 ms | 992.85 ms | 477.75 ms | 260 tokens |
| FP16 | 4 thread (multi-thread) | K = 3 | 92.45 ms | 139.08 ms | 245.27 ms | 104.05 ms | 62 tokens |
| FP16 | 4 thread (multi-thread) | K = 5 | 100.30 ms | 116.98 ms | 145.23 ms | 102.78 ms | 80 tokens |
| FP16 | 4 thread (multi-thread) | K = 9 | 135.04 ms | 149.11 ms | 200.65 ms | 137.32 ms | 116 tokens |
| FP16 | 4 thread (multi-thread) | K = 17 | 211.50 ms | 261.49 ms | 556.34 ms | 231.07 ms | 188 tokens |
| FP16 | 4 thread (multi-thread) | K = 25 | 331.93 ms | 451.44 ms | 694.71 ms | 357.75 ms | 260 tokens |
<!-- END GENERATED: walkthrough_latency -->

---

## 4. Key architectural interventions

1. Semantic label glossary (`core/banking_glossary.py`):
   Replaced shorthand tags with natural language definitions (such as `"Inquire about whether a newly ordered debit or credit card has arrived or when it will arrive in the mail"`).
2. Proper scoring rule selection (`scripts/train.py`):
   Selected model checkpoints by tracking validation negative log-likelihood (NLL) with linear warmup and cosine learning rate decay, preventing probability overconfidence in later epochs.
3. Composite Brier score loss:
   Trained using composite loss ($\mathcal{L} = \mathcal{L}_{\text{CE}} + 1.0 \times \mathcal{L}_{\text{Brier}}$) to penalize overconfident probabilities during backpropagation.
4. Post-hoc temperature scaling (`scripts/evaluate.py`):
   Fitted positive scalar temperature $T = \exp(\theta)$ on held-out calibration logits via L-BFGS optimization.

---

## 5. Production policy gating verification

Testing deterministic policy gating rules against incoming queries:

1. Definitive transaction dispute:
   Input: *"Cardholder was charged $120.00 twice on credit card at CloudHost Ltd."*
   Output: Selects dispute intent with high calibrated confidence (> 90%).
   Policy: Clears the 85% automated threshold and proceeds to dispatch.

2. Ambiguous customer ticket:
   Input: *"Customer message: 'I need urgent assistance with my recent transaction.' No details provided."*
   Output: Confidence falls below the 85% threshold.
   Policy: Routed to a human supervisor review queue with an audit trail.

3. Out-of-scope input:
   Input: *"The 2024 total solar eclipse crossed North America..."*
   Output: Explicit abstention (`__insufficient_evidence__`) with high confidence.
   Policy: Routed to standard triage rather than executing an erroneous banking workflow.

---

## 6. Test suite verification

The test suite covers:
- `tests/test_calibration.py` (Brier score loss, analytic gradients, L-BFGS optimization, ECE calculation, bootstrap confidence intervals).
- `tests/test_dag.py` (Topological DAG execution, cycle detection, parent abstention blocking).
- `tests/test_engine_causal.py` (Causal slot evaluation, multitoken rejection, boundary assertions).
- `tests/test_engine_encoder.py` (ModernBERT GLiClass single-pass evaluation, concentration statistics).
- `tests/test_export_parity.py` (PyTorch versus ONNX Runtime numerical parity across v2, v2_fp16, and v1 bundles).
- `tests/test_browser_parity.py` (Exact prompt construction byte parity, presets glossary parity, and browser Python inference parity).
- `tests/test_docs_fresh.py` (Verifies that documentation matches committed JSON receipts without manual drift).
- `tests/test_formatting.py` (Prompt formatting, capacity guards, Noul semantics requirements).
- `tests/test_primitives.py` (Pydantic v2 schemas: Choice, Score, Noul, immutability, validator guards).
- `tests/test_tabular.py` (TableCompactor sliding windows, continuation records, lossless roundtrips).
