# Agent context: Verdict 2.0

Read this file first. It is the entry point for any agent or person picking up this repository cold.

This is a private working repository. The public v1 repository is a separate one,
`Heman10x-NGU/Verdict-open-jev`, and nothing here should change it.

## What this project is

Verdict evaluates typed decision schemas (Choice, Score, Noul) over a workflow state in a single
non-autoregressive forward pass, returning calibrated probabilities rather than generated text.
Verdict 2.0 is a rebuild aimed at the `LocalLLaMA/typed-decisions` benchmark.

## Current status, 2026-09-19

| Phase | State |
|---|---|
| Phase 0, diagnose the v1 baseline | NOT DONE. Script exists at `scripts/analysis/phase0_diagnose.py`. Must run on the Mac, where `artifacts/v2` lives. |
| Phase 1, harness and reference floors | Partly done. `reports/reference_floors.json` is committed. |
| Phase 2, backbone decision | DONE. Settled in favour of ModernBERT. See plan section 7. |
| Phase 3, training pipeline | Code written and smoke tested on CPU. No real training run has happened yet. |
| Phase 4, synthetic data | Not started, and optional. Has a kill criterion. |
| Phase 5, test read and report | Not started. The test split has not been read by any Verdict 2.0 model. |

No Verdict 2.0 accuracy number exists yet. Do not cite one.

## Read in this order

1. `plans/Claude/IMPLEMENTATION_PLAN.md` is the specification. Sections 1, 2, and 7 are the ones
   that change decisions.
2. `plans/Claude/EXPLANATION.md` is the same material in plain language, with the time budget.
3. `RUNBOOK.md` is the operational sequence for the training laptop.
4. `plans/Claude/STRATEGY-MODELS-AND-COST.md` covers model choice, competitor status, and cost.

`plans/Gemini/` holds an earlier draft that was graded D+ and superseded. Read it only to understand
what was rejected and why. Do not execute it.

The numbered documents at the repository root (`00-` through `09-`) are v1 history. They predate
every finding below and several of their claims are now known to be wrong. Treat them as an archive.

## Facts that are measured, not assumed

Do not re-derive these, and do not contradict them without new measurements.

1. The benchmark is small and closed. 1,200 train cases and 400 test cases, 5 questions each,
   exactly 20 distinct question schemas shared across both splits, option cardinality in {2, 4, 5},
   state length p50 237 tokens and max 491.
2. Labels are soft teacher-panel distributions. Mean top probability 0.659. The panel reaches
   unanimous argmax on 59.4% of test decisions. On 1.55% of items, `label` differs from
   `argmax(probabilities)`, so distribution matching alone caps accuracy at 98.45%.
3. Brier and ECE cannot both be hit. The harness scores Brier against the soft panel distribution
   and ECE against hard correctness. At Brier 0.105 the best reachable ECE is about 0.091. At
   ECE 0.040 the best reachable Brier is about 0.150. This is why the model has two output channels.
4. Reference floors on the test split, in `reports/reference_floors.json`: uniform random about 0.27,
   per-question majority label 0.4835, TF-IDF plus logistic regression about 0.65 with an ECE near
   0.03, gold-distribution oracle 0.9845.
5. Verdict 1.0 scores 0.2610, which is below uniform random, with Brier 0.5851 against uniform's
   0.2433. Being reliably worse than chance points at a wiring defect. Phase 0 exists to find it.
6. Competitor numbers in circulation are unreliable. The Jev row of 0.727 is a hardcoded literal in
   a Laya notebook, not a measurement. Laya's real published typed-decisions card reads accuracy
   0.766, Brier 0.066, ECE 0.214, which is exactly the tradeoff in point 3. Kev has never been
   evaluated on this benchmark.

## What we take from Laya and from Kev

Both reference repositories are vendored in this one, at `Laya/` and `Kev/`. Read them locally.
Nothing needs downloading.

Taken from Laya (ModernBERT-large encoder, 0.766 accuracy on this benchmark):

| Idea | Where it lives now |
|---|---|
| Marker-pointer layout, `[CLS] type question: instructions [SEP] [MASK]opt0 [MASK]opt1 ... [SEP] state [SEP]` | `verdict2/data.py::build_item` |
| Scoring an MLP over each `[MASK]` hidden state to get one logit per option | `verdict2/model.py::VerdictModel.scorer` |
| Training against soft teacher distributions with proper scoring rules | `verdict2/losses.py::decision_loss` |
| Ranked probability score for ordinal Score questions | `verdict2/losses.py::ranked_probability_score` |
| Per-bucket temperature scaling | `verdict2/train.py::fit_temperature`, keyed on (qtype, cardinality) |
| The `act_head` idea of reading confidence off distribution shape | Extended into `verdict2/model.py::CorrectnessHead` |

Taken from Kev (Qwen2.5-0.5B, block-causal branch mask, 104-line model):

| Idea | Where it lives now |
|---|---|
| Sanitizing caller text so it cannot forge a delimiter or marker token | `verdict2/data.py`, every user string has the mask token stripped |
| Permutation-KL training, symmetric KL between two option orderings | `verdict2/losses.py::permutation_kl`, wired in `train.py` behind `--perm_kl` |
| The full permutation measurement suite, not only a flip rate | `verdict2/evaluate.py::permutation_stability` |
| Loud failure on truncation rather than a silent drop | `verdict2/data.py::load_items` counts and warns |
| Cross-entropy plus an ordinal regularizer rather than a cumulative link model | `verdict2/losses.py` |

Deliberately not taken from Kev: its abstention and distractor augmentation (`p_none`,
`p_distract`). This benchmark has no abstention option, and Kev's own measurements show its
"none of the above" behaviour is correct anyway, so the premise for that work was wrong.

Deliberately not taken from Laya: re-encoding once per question. Laya's latency is linear in
question count (38ms for 1, 156ms for 10, 721ms for 50). Batching the five questions of a case in
one pass is where our latency win comes from.

## How we beat each of them

| Competitor | Their weakness, measured | Our answer |
|---|---|---|
| Laya, 0.766 accuracy | ECE 0.214, the worst calibration of the three | Two output channels, so the distribution can match the panel while confidence tracks hit rate |
| Jev 1.13.0 | ECE 0.144 and 710ms per case | Same two-channel design, plus a 150M to 396M local encoder |
| Kev | 7.41% argmax flip rate under option reordering | `--perm_kl` training plus the same measurement, reported against Kev's numbers |

Beating all three at once means winning on calibration and order stability, not on raw accuracy.
Accuracy above roughly 0.78 runs into teacher label noise, per fact 2 above.

## Invariants

1. No open-ended text generation. The model evaluates typed primitives only.
2. Never report one number as if it answered both calibration questions. Every report carries both
   `ece_distribution` (the stock harness metric on `max p`) and `ece_confidence` (the correctness
   head). Publishing only the second is metric gaming.
3. Always publish the reference floors from point 4 alongside any headline number.
4. The test split is a vault. Temperature, the conformal quantile, the correctness head, early
   stopping, and every hyperparameter choice are fitted on the train split's calib or dev folds.
   `verdict2/evaluate.py` gates the test read behind an explicit checklist. Count your reads.
5. Deterministic code controls policy. The model supplies judgment; thresholds, dispatch, and audit
   trails stay in ordinary software.
6. Single-pass inference. No autoregressive token loops for decision primitives.

## Two invariants from the v1 AGENTS.md that are now retired

State them explicitly, because earlier drafts and the archived root documents still assert them.

1. The old rule required every schema to carry an `insufficient_evidence` or `none_of_the_above`
   option. `LocalLLaMA/typed-decisions` has no abstention option anywhere. Injecting a dummy option
   corrupts the label space and breaks alignment with the gold distributions. Do not add one. An
   abstention primitive is a separate future workstream with its own benchmark.
2. The old rule forbade committing `plans/`. That applied when this work lived in a public
   repository. This repository is private and the plans are committed on purpose, so an agent on
   another machine can continue from them. Do not untrack them.

## Code layout

| Path | Purpose |
|---|---|
| `verdict2/data.py` | Sequence construction, soft targets, case-level fit/calib/dev splits |
| `verdict2/model.py` | Encoder backbone, marker-pointer head, correctness head |
| `verdict2/losses.py` | Soft cross-entropy plus Brier plus RPS, routed by question type |
| `verdict2/metrics.py` | Harness-compatible metrics plus the second ECE channel |
| `verdict2/train.py` | Training, temperature fitting, correctness-head fitting. Never reads test |
| `verdict2/evaluate.py` | The single final test read, behind an anti-leak gate |
| `scripts/analysis/phase0_diagnose.py` | Four controls on the v1 baseline |
| `scripts/analysis/reference_floors.py` | Regenerates the baseline floors |
| `core/` | v1 engine and Pydantic primitives. `core/primitives.py` is current and worth keeping |

## Hardware

Training target is a GTX 1660 Ti, 6 GB with about 5.5 GB usable, compute capability 7.5. bf16 and
FlashAttention-2 both require compute capability 8.0 and are unavailable. Use `--fp16` and PyTorch
SDPA. Full fine-tuning ModernBERT-large needs `--optim8bit --grad_checkpoint` to fit. The arithmetic
is in `RUNBOOK.md` section 0.

## Gates

Do not skip these in either direction, meaning do not proceed past a failed gate and do not stall on
a passed one.

- Phase 0 must produce a written diagnosis before new training is trusted.
- Dev accuracy at or above 0.700 on ModernBERT-base before moving to large. Below that, the problem
  is the data path or the objective, and a bigger backbone will not rescue it.
- Dev accuracy at or above 0.760 and dev Brier at or below 0.120 before reading the test split.
- Ship floor is 0.700 test accuracy. Below it, the honest outcome is to ship the TF-IDF baseline and
  say so.

## Writing and code style

1. Google developer style: active voice, second person, present tense, sentence-case headings,
   serial commas, and inline parenthetical explanations for jargon.
2. No em dashes. Use colons, commas, or parentheses.
3. No bold bullet labels.
4. No negative parallelisms of the form "not X, but Y".
5. Python is typed with PEP 484 annotations, and Pydantic schemas at system boundaries.

## Git

Never merge. Push to feature branches and open pull requests. The founder merges everything.
Never push to `main` on any public repository.
