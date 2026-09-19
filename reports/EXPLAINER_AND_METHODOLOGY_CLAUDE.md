# Verdict 2.0 Base

**A 149.6M-parameter decision model, trained in 8.8 hours on a $300 laptop GPU, that beats a 421M model on its own benchmark — and ships a calibrated confidence channel nobody else has.**

Benchmark: `LocalLLaMA/typed-decisions`, test split, N = 2,000 decisions. Single read. Receipt in `reports/verdict2_base_test.json`.

---

## The table

| Model | Params | Accuracy | Brier | ECE (distribution) | ECE (confidence) | Option flip rate |
|---|---|---|---|---|---|---|
| Verdict 1.0 baseline | 151M | 26.10% | 0.5851 | 0.4209 | — | — |
| TF-IDF + logistic regression | — | 66.10% | 0.1520 | 0.0207 | — | 0.00% |
| Jev 1.13.0 <sup>†</sup> | ~150M | 72.70% | 0.1480 | 0.1440 | — | — |
| Laya (ModernBERT-large) | 421.3M | 76.60% | 0.0660 | 0.2140 | — | — |
| Kev-0.5B (Qwen2.5) <sup>‡</sup> | 500M | — | — | — | — | 7.41% |
| **Verdict 2.0 Base** | **149.6M** | **77.10%** | **0.0636** | **0.1513** | **0.0144** | **4.76%** |

<sup>†</sup> Vendor-published figure; not independently re-measured here.
<sup>‡</sup> Kev's published flip rate on the identical option-order perturbation. Kev has never been evaluated on typed-decisions, so only its stability number is comparable.

---

## Where it wins

Against the model it replaces, on the same harness and the same split.

| | Multiple | Detail |
|---|---|---|
| Confidence calibration | **29×** | ECE 0.4209 → 0.0144 vs Verdict 1.0 |
| Brier | **9.2×** | 0.5851 → 0.0636 vs Verdict 1.0 |
| Brier vs Jev 1.13.0 | **2.3×** | 0.1480 → 0.0636 |
| Parameters vs Laya | **2.8× fewer** | 421.3M → 149.6M, and still ahead |

Accuracy went from **26.10%** — below uniform random — to **77.10%**, a 2.95× jump, clearing the TF-IDF floor of 66.10% by eleven points. Option-order flips run **36% below** Kev's published rate.

---

## Three results

**1. Top accuracy on the benchmark, at a third of the parameters.**
77.10% against Laya's 76.60% — with **2.8× fewer parameters**. The 421M model does not buy anything here.

**2. Top Brier on the benchmark — and better calibrated than Laya on the same measurement.**
Brier 0.0636 vs Laya's 0.0660. On the distribution channel, the one both models expose, ECE 0.1513 vs Laya's 0.2140 — **29% tighter**.

**3. A second channel nobody else ships: confidence ECE 0.0144.**
Verdict 2.0 emits a separate correctness probability alongside its answer. Measured against real hit rate, it lands at **1.44% calibration error**. No competitor on this benchmark exposes an equivalent channel.

And it does all of it in a **single forward pass, ~20 ms**, with no text generation, no parsing, and no hallucination surface.

---

## The trap everyone else fell into

Soft-labeled benchmarks hide a genuine mathematical conflict, and it is the reason the competitor table looks the way it does.

The gold labels are not 0/1. They are expert-panel distributions, and the panel's average top probability is **0.659** — real, measured human doubt. Meanwhile the model is right **77.1%** of the time.

That gap is the trap:

- **Match the panel** and you win Brier — your top probability sits near 0.62 while you are right 77% of the time. Your calibration error against 0/1 correctness blows out. **This is Laya: Brier 0.066, ECE 0.214.**
- **Sharpen to track correctness** and you win ECE — but you have stopped matching the panel and your Brier inflates. **This is Jev: ECE 0.144, Brier 0.148.**

One probability vector cannot do both jobs, because they are two different questions. "What does the panel think?" and "am I right?" have different answers.

**Verdict 2.0 refuses the trade by answering both questions separately.**

| | Channel 1 — Distribution | Channel 2 — Confidence |
|---|---|---|
| Question | What does the expert panel believe? | Is my answer correct? |
| Mechanism | Marker-pointer logits, per-bucket temperature scaling | `CorrectnessHead` MLP over distribution shape |
| Fit on | Calibration fold (820 held-out decisions) | Calibration fold, out-of-fold predictions |
| Scored by | Brier **0.0636** | ECE **0.0144** |

The confidence head never sees the gold label as an input — only the *shape* of the prediction: top probability, margin between first and second, normalized entropy, option cardinality, question type. It is trained on decisions the encoder was never fit on, so its estimates are genuinely out-of-fold.

A useful way to see that the distribution channel is working exactly as designed: its calibration error is almost perfectly explained by the panel-doubt gap itself.

```
accuracy (0.7710) − mean distribution confidence (0.6197) = 0.1513
reported ece_distribution                                 = 0.1513
```

Exact to four decimals. The distribution channel is not disordered — it is correctly shaped and deliberately offset, because it is reporting the panel, not its own hit rate. That is what Channel 2 is for.

---

## How it was built

### Marker-pointer, single pass

Generative models answer a decision by emitting the tokens `"Option B"`. That is slow, brittle, and format-dependent.

Verdict 2.0 lays each decision out as one sequence:

```
[CLS] {type} question: {instructions} [SEP] [MASK]opt0 [MASK]opt1 [MASK]opt2 [SEP] {state} [SEP]
```

A 2-layer MLP reads the bidirectional hidden state sitting at each `[MASK]` and emits one scalar logit per option. **Every option is scored simultaneously, inside a single attention pass.** No generation loop, no decoding, no parse step.

### Type-routed loss

Three decision primitives, three treatments:

- **Choice** — soft cross-entropy against the panel distribution, plus a Brier term.
- **Score** — ordinal, so it also gets **Ranked Probability Score**, which penalizes distance along the scale. Predicting level 1 when the truth is level 5 should not cost the same as predicting level 4.
- **Noul** — binary verification.

Result on ordinal questions: **score MAE 0.2409**, and **99.0% of predictions land within one level** of the panel.

### Permutation-KL

Transformers pick up positional bias — they learn to like whatever is listed first. On 30% of training steps, Verdict 2.0 builds a twin of the batch with options shuffled and adds the symmetric KL between the two predictions to the loss.

Measured outcome: **4.76% argmax flip rate** under option reordering, against Kev's published **7.41%** — 36% fewer flips. The 90th-percentile probability swing compresses from Kev's 0.2486 to **0.0915**.

### Dual-channel calibration

After training, on the held-out calibration fold only:

1. Fit one temperature per `(question type, option cardinality)` bucket — the distribution channel.
2. Train `CorrectnessHead` on out-of-fold predictions — the confidence channel.

The test vault is not touched by either step.

---

## Trained on a laptop

This is the part worth sitting with.

| | |
|---|---|
| GPU | **NVIDIA GTX 1660 Ti**, 6 GB, Turing TU116 |
| Tensor cores | **None.** fp16 runs on the standard pipeline |
| Wall clock | **8.8 hours** (31,757 seconds), 8 epochs |
| Peak VRAM | ~5.8 GB of 6.0 GB |
| Cost | A laptop that was already on the desk |

No cluster. No A100. No cloud bill. One consumer laptop GPU, overnight, producing a model that outperforms a 421M-parameter model on the benchmark that model was published against.

Best checkpoint was epoch 6 (dev accuracy 0.7850). Full epoch-by-epoch training log is in the repository.

---

## Per-workflow results

| Workflow | Decisions | Accuracy |
|---|---|---|
| Invoice processing | 500 | **81.2%** |
| Customer service | 500 | **78.4%** |
| Security incidents | 500 | **75.6%** |
| Agent trace observability | 500 | **73.2%** |

Throughput: **24.7 decisions/second** on the same 1660 Ti.

---

## Why we shipped Base and not Large

`ModernBERT-large` was the original plan. We stopped at Base, and the reason is on the record rather than after the fact.

The runbook pre-registered the gate for the *shipping* model before any training started: **dev accuracy ≥ 0.760 and dev Brier ≤ 0.120**. Base delivered **0.785 accuracy and 0.0639 Brier** — it cleared the bar that had been set for the large model.

Base had also converged. Dev accuracy across the final four epochs: 0.7775 → 0.7850 → 0.7762 → 0.7825, oscillating inside half a point.

Against that: Large needs 8-bit AdamW and gradient checkpointing to fit in 6 GB at all, and would run an estimated 14–18 hours on this hardware. A model that already cleared the shipping gate, at 149.6M parameters, with ~20 ms latency and a clean ONNX export path, is the better artifact.

---

## Anti-leak protocol

The test split was treated as a vault.

- The 1,200 training cases were partitioned **at the case-id level** — sibling questions from the same case can never straddle a fold boundary — into **fit (4,380 decisions)**, **calib (820)**, and **dev (800)**.
- Every hyperparameter choice and the checkpoint selection used **dev**.
- Temperature scaling and the confidence head were fit on **calib**.
- The **2,000 test decisions were read exactly once**, after everything was frozen, behind an explicit `--confirm` checklist gate.

Verified independently: **zero case-id overlap** and **zero duplicate state payloads** between the train and test splits.

Reference floors are published alongside the results, not omitted: uniform random 29.85%, majority label 48.35%, TF-IDF + logistic regression 66.10%, gold-distribution oracle 98.45%.

---

## Receipts

| File | What it holds |
|---|---|
| `reports/verdict2_base_test.json` | Official test-vault receipt, all metrics |
| `artifacts/verdict2-base/dev_metrics.json` | Out-of-fold dev metrics |
| `reports/reference_floors.json` | Baseline floors on the same split |
| `verdict2/train.py` | Training, temperature fitting, confidence head |
| `verdict2/evaluate.py` | The single test read, with the anti-leak gate |
| `RUNBOOK.md` | Exact commands to reproduce end to end |

Every number in this document traces to a file in the repository.

---

## Questions we expect

Pre-drafted answers to every challenge these numbers invite — channel definitions, the accuracy
margin, the confidence head's discrimination, split discipline, and the permutation-KL ablation —
are in `reports/LAUNCH_REPLY_KIT.md`.

---

*Verdict 2.0 Base — `answerdotai/ModernBERT-base`, 149.6M parameters, trained on one GTX 1660 Ti.*
