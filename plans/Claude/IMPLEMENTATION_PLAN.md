# Verdict 2.0 implementation plan (Claude revision)

Status: supersedes `plans/Gemini/IMPLEMENTATION_PLAN.md`. Every number in section 1 was measured
on this machine against the cached copy of `LocalLLaMA/typed-decisions` and against the vendored
`Laya/` and `Kev/` trees. Reproduction commands are in section 11.

## 0. What changed and why

The Gemini draft specifies a system for a benchmark that does not exist. It assumes 4,096-token
state contexts, 12,000 schema topologies, option cardinality up to 64, and a hostile abstention
shortcut inherited from Kev. The actual benchmark has 237-token median states, 20 fixed schemas,
cardinality in {2, 4, 5}, and no abstention option at all. It also sets a joint Brier and ECE
target that no predictor can satisfy, and routes training through a PyTorch kernel that has no
backward pass on the hardware the plan names.

This revision keeps four ideas from that draft (permutation-invariant option attention, typed loss
routing, ordinal treatment of Score, conformal prediction sets), fixes the targets, replaces the
execution and training paths with ones that run, and adds the piece both drafts were missing:
a confidence channel that is separate from the reported probability distribution.

## 1. Ground truth: what the benchmark actually is

All figures below are measured, not quoted.

### 1.1 Dataset shape

| Property | Measured value |
|---|---|
| Train split | 1,200 cases, 6,000 decisions |
| Test split | 400 cases, 2,000 decisions |
| Questions per case | Exactly 5, always |
| Workflows | 4, each 300 train / 100 test cases |
| Distinct (workflow, question id, type) schemas | 20, closed set, identical across splits |
| Question types (test) | 600 choice, 600 noul, 800 score |
| Option cardinality | choice K in {4, 5}, noul K = 2, score K in {4, 5}. Maximum K is 5 |
| State length (Qwen-class BPE) | min 80, p50 237, p95 348, max 491 tokens |
| Per-question branch length | p50 57, p95 78, max 79 tokens |
| Packed sequence, state plus all 5 branches | p50 516, p95 656, max 799 tokens |

There are no dynamic schemas, no high-cardinality option sets, no long documents, and no
abstention candidates. The task is closed-set classification over 20 known label spaces.

### 1.2 Labels are soft teacher-panel distributions

Each gold entry carries `label`, `confidence`, a full `probabilities` distribution, and for score
questions an expected `score`. A sibling `label_agreement` field records panel agreement.

* Mean gold top probability on test is 0.659 (p10 0.450, p50 0.617, p90 0.947). The panel is
  genuinely split on most questions.
* The panel reaches unanimous argmax on 59.4% of test decisions and 64.65% of train decisions.
* On 1.55% of test decisions `label` differs from `argmax(probabilities)`. A model that perfectly
  reproduces the gold distribution therefore scores 98.45% accuracy, not 100%.
* 3.8% of test score distributions are not unimodal.

This last point matters for section 4.4: a cumulative link model with one latent scalar can only
emit unimodal distributions, so it cannot represent 3.8% of the score targets.

### 1.3 What the harness actually measures

`scripts/evaluate_verdict_baseline.py` is the scoring contract. Read it as the specification,
because two of its metrics do not mean what their names suggest.

* Accuracy is top-1 exact match against `label`, over all 2,000 decisions.
* Brier is `sum_k (p_k - g_k)^2` against the soft gold distribution `g`, averaged over the 1,200
  choice and noul decisions only. Score questions contribute no Brier. This is a distribution
  distance, so it is minimized by reproducing the teacher panel, and it reaches 0 for a model that
  matches it exactly.
* ECE bins `max_k p_k` against hard 0/1 correctness. It is minimized by reporting a confidence
  equal to the probability of being right.
* Score MAE compares the predicted expected level to the gold expected `score`.

Brier rewards reporting the panel's uncertainty. ECE rewards reporting your own hit rate. Those are
different quantities, and on this dataset they pull in opposite directions (section 2).

### 1.4 Where Verdict 1.0 actually stands

| Predictor | Accuracy | Brier | ECE |
|---|---|---|---|
| Uniform over the option set | 0.2690 | 0.2433 | 0.0485 |
| Verdict 1.0 (151M, measured, 3 runs) | 0.2610 | 0.5851 | 0.4209 |
| Per-(workflow, question) majority label from train | 0.4835 | n/a | n/a |
| TF-IDF character n-grams plus logistic regression | 0.6505 | 0.1445 | 0.0314 |
| Gold-distribution oracle | 0.9845 | 0.0000 | 0.3256 |

Verdict 1.0 scores below uniform random on accuracy and 2.4x worse than uniform on Brier. A model
that had merely learned nothing would sit at 0.269 with Brier 0.243. Scoring 0.261 with Brier 0.585
means the system is confidently and systematically wrong, which is the signature of a wiring defect
(option identity misalignment, description and id swap, or a label mapping inversion), not of a
weak architecture. Section 10 makes diagnosing this a gate before any training starts.

The TF-IDF row is the number that should reframe the project. Thirty seconds of scikit-learn on the
train split already beats the ECE target in the brief, lands within 0.04 of the Brier target, and
reaches 65.05% accuracy. Any plan costing 15 to 19 hours of compute has to clear that bar by a wide
margin to have been worth running.

### 1.5 The competitor table in the brief does not survive checking

| Claim in the brief | What the vendored evidence shows |
|---|---|
| Jev 1.13.0: 72.70% / 0.1480 / 0.1440 / 710 ms | A hardcoded constant in `Laya/github-laya/notebooks/laya_eval_typed_decisions_colab.ipynb` cell 14. No one in this repo measured it. |
| Laya: 76.60% | A string in the publish cell of the Kaggle fine-tune notebook ("Publishing 0.766 accuracy..."), for Laya fine-tuned on the 1,200-case train split. Laya's own shipped `eval/results.json` is a different benchmark entirely (23,024 questions, task families like sentiment and emotion) and reports 0.753 in-task, 0.651 zero-shot. |
| Laya Brier 0.1190, ECE 0.0810 | 0.119 is the ModernBERT-base row of that same hardcoded table, not Laya. Laya's shipped overall Brier is 0.308. |
| Laya latency ~156 ms | Laya's own latency table: 38.4 ms for 1 question, 156.0 ms for 10, 721.4 ms for 50. The benchmark uses 5 questions per case. Laya re-encodes per question, so its cost is linear in question count. |
| Kev-0.5B: 78.10% / 0.1120 / 0.0650 | 0.78125 is Kev's macro accuracy on Kev's own suite (AG News, Banking77, BoolQ, MNLI, SST-5, Yelp). 0.0653 is `accuracy_calibration.ALL` from a third evaluation (n=1,350). Kev's measured Brier on its own dev suite is 0.3021. Kev has never been evaluated on typed-decisions. |
| Kev latency ~38 ms | Kev's README states "serves a six-question request in ~160 ms" on an Apple M5, and its demo output shows `latency_ms: 162`. The 38 ms figure is Laya's single-question number. |

The sub-35 ms latency target descends from that last misattribution. Before adopting any of these
numbers as a target, either re-run the competitor on this harness or label the number as a vendor
claim on a different benchmark.

## 2. Feasibility: the Brier and ECE targets are mutually exclusive

Binned ECE obeys `ECE >= |accuracy - mean confidence|`, because
`sum_b (n_b/n)|acc_b - conf_b| >= |sum_b (n_b/n)(acc_b - conf_b)|`.

Mean gold top probability is 0.659. A model whose reported distribution matches the panel therefore
reports mean confidence near 0.659 while being right far more often, so it books a large ECE. A model
sharpened until its confidence tracks its hit rate no longer matches the panel, so it books a large
Brier.

Sharpening the gold distribution itself with a temperature (an upper bound on any real model, since
this predictor is 98.45% accurate) traces the frontier:

| Sharpening T applied to gold | Accuracy | Brier | ECE | Mean confidence |
|---|---|---|---|---|
| 1.00 | 0.9845 | 0.0000 | 0.3256 | 0.659 |
| 0.30 | 0.9845 | 0.0698 | 0.1260 | 0.863 |
| 0.20 | 0.9845 | 0.1028 | 0.0907 | 0.906 |
| 0.15 | 0.9845 | 0.1233 | 0.0684 | 0.928 |
| 0.10 | 0.9845 | 0.1466 | 0.0458 | 0.950 |
| 0.05 | 0.9845 | 0.1718 | 0.0247 | 0.973 |

At Brier 0.105 the best reachable ECE is about 0.091. At ECE 0.040 the best reachable Brier is about
0.150. A direct constrained optimization agrees: holding a predictor to `mean confidence >= accuracy - 0.04`
and minimizing Brier by Euclidean projection onto the simplex gives a floor of 0.254 at 81.5%
accuracy and 0.174 even at 99% accuracy.

Conclusion: `accuracy >= 0.815 and Brier <= 0.105 and ECE <= 0.040` is unreachable by any predictor.
Reporting all three from one probability vector cannot be done. Section 4.4 resolves this by
reporting two quantities instead of one.

## 3. Revised targets

Targets are tiered so the project has an honest stopping rule.

| Metric | Floor (ship-blocking) | Target | Stretch |
|---|---|---|---|
| Top-1 accuracy | 0.700 | 0.790 | 0.815 |
| Brier, reported distribution | 0.140 | 0.105 | 0.090 |
| ECE on reported distribution `max p` | report it, no gate | report it | report it |
| ECE on the correctness channel | 0.060 | 0.040 | 0.030 |
| Conformal coverage at alpha = 0.05 | 0.93 empirical | 0.95 | 0.95 with mean set size < 1.8 |
| Score MAE | 0.500 | 0.420 | 0.380 |
| Latency p50, 5 questions, M-series Max, fp16 | 200 ms | 120 ms | 90 ms |
| Latency p50, 5 questions, T4 fp16 | 80 ms | 45 ms | 25 ms with INT8 |

The 0.700 floor is set 5 points above the TF-IDF baseline. If the trained model cannot clear the
floor, the correct outcome is to ship the classical baseline and say so.

Latency: one packed forward at 516 tokens through Qwen2.5-0.5B without the vocabulary head costs
about 370 GFLOP. Sub-35 ms on a laptop implies above 10 TFLOP/s of sustained throughput at batch 1,
which Apple Silicon does not deliver for this shape. If sub-35 ms is a hard product requirement,
that requirement selects the encoder tier in section 7, not the decoder.

## 4. Architecture

### 4.1 Execution: one packed sequence with an additive block mask

Encode state and all Q question branches into a single sequence and control visibility with an
additive float mask. Token `i` attends to token `j` when `j <= i` and (`seg[j] == 0` or
`seg[j] == seg[i]`), where segment 0 is the state.

This is the mechanism Kev already ships in `Kev/github-kev/kev/model.py`, and Kev measured both of
the properties the Gemini draft wanted to prove:

* `packed_vs_separate`: max absolute probability difference between packed and separately-encoded
  questions is 3.70e-06, and packing is 2.02x faster.
* `isolation`: mean probability assigned to a sibling-only answer is 0.030430 when the sibling is
  present and 0.030430 when it is absent, identical to six decimal places.

Reject the two-phase decoupled prefix-KV engine from the Gemini draft for this workload:

* Its premise is a 4,096-token state and a 48 MiB cache. Measured p95 state length is 348 tokens,
  which is a 4.08 MiB cache. The problem it solves does not occur here.
* Section 1.2 of that draft promises "zero attention mask overhead" while section 1.6 requires
  masking options from each other. Those cannot both hold. Once a custom mask is needed, a single
  packed block mask is the simpler construction.
* `expand()` on the batch dimension of a KV cache produces a zero-stride view. The Transformers
  cache path concatenates onto cached tensors, which materializes the copy anyway. At 4 MiB times
  5 branches the copy is free, so the optimization buys nothing.

Keep prefix caching as a later, optional API feature for the case where one state is queried across
separate calls over time. Do not put it on the critical path.

### 4.2 Option isolation for exact permutation invariance

This is the one architectural idea in the Gemini draft that is a real improvement, and it is worth
implementing. Extend the mask so option spans within a branch do not attend to each other:

```
attend(i, j) iff j <= i
               and (seg[j] == 0 or seg[j] == seg[i])
               and not (opt[i] != 0 and opt[j] != 0 and opt[i] != opt[j])
```

where `opt[t]` is the 1-based option index of token `t` inside its branch, or 0 for the branch
instruction and the decide token. Option tokens see the state and the instruction. They do not see
sibling options. The decide token sees everything in its branch.

Kev measured a 7.41% argmax flip rate under option reordering and a p90 probability spread of 0.249,
using a symmetric-KL regularizer rather than a mask. The mask makes invariance exact and costs
nothing at inference. Verify it as a hard test, not as a metric: assert that logits are bitwise
identical (within 1e-5) under 20 random option permutations on 100 test cases.

One caveat to check during implementation: masking options from each other removes the model's
ability to contrast candidates against each other inside the forward pass. Run the ablation
(masked versus unmasked) on the dev fold before committing. If unmasked wins on accuracy by more
than 1.5 points, keep causal option attention and fall back to Kev's permutation-KL plus a reported
flip rate. Do not trade 1.5 accuracy points for an invariance property nobody asked for.

### 4.3 Position assignment

Branch positions restart at `len(state)` so every branch sits at the same rotary distance from the
state. This is the Gemini draft's section 1.4, and it is already what Kev's `encode()` does
(`pos += list(range(len(S), len(S) + len(br)))`). Implement it, and do not present it as novel.

### 4.4 Heads: separate the distribution from the confidence

This is the central change from both drafts and the answer to section 2.

Head 1, distribution pointer. For each question, a pointer head reads the decide-token hidden state
and each option's closing-token hidden state, and emits K logits:

```
z_k = (W_K h_opt_k) . (W_Q h_decide) / sqrt(d_proj),  d_proj = 256
```

Use FP32 for the projection and the softmax. Keep the plain scaled dot product from Kev rather than
the cosine plus `1/sqrt(ln K)` scaling from the Gemini draft. With K in {2, 4, 5}, that correction
varies only between 12.01 and 7.87 across the entire benchmark, so it cannot matter here, and its
stated justification (FP16 accumulator overflow above 65,504) cannot occur for a cosine bounded in
[-1, 1] scaled by 10. Revisit cardinality scaling only if a later workstream introduces K above 16.

Head 1 output `p_hat` is what gets reported as the probability distribution. It is scored by Brier,
soft accuracy, total variation, and KL. It is trained to match the teacher panel.

Head 2, ordinal term for Score. Keep the free-form pointer over levels so multimodal targets stay
representable, and add a ranked probability score term on the CDF:

```
RPS = sum_{m=1}^{K-1} ( cumsum(p_hat)_m - cumsum(g)_m )^2 / (K - 1)
```

Do not adopt the cumulative link model as the primary parameterization. A single latent scalar with
monotone thresholds forces unimodality, and 3.8% of test score targets are not unimodal. Run the
cumulative link as a recorded ablation. If it wins on Score MAE without losing Brier, promote it.

Head 3, correctness. A small MLP over distribution-shape features predicts the probability that the
argmax matches the teacher's `label`:

```
features = [ p_(1), p_(1) - p_(2), H(p)/ln K, 1/K, onehot(qtype) ]   # 8 dims
q_hat    = sigmoid( MLP(features) )                                  # 2 layers, width 64
loss     = BCE( q_hat, 1[argmax p_hat == label] )
```

Train head 3 on out-of-fold predictions only (section 5.1), never on data head 1 was fit on, or it
will learn head 1's training-set optimism. `q_hat` is the number reported as `confidence` and the
number ECE is computed against. Laya already carries a near equivalent (`act_head` over
`[top1, margin, entropy, k]`), so the construction is proven in this model family.

Report both ECE figures in every benchmark run, clearly labelled: `ece_distribution` computed on
`max p_hat` as the stock harness does, and `ece_confidence` computed on `q_hat`. Never quote the
second without the first. Anyone comparing against Laya or Kev needs the stock number.

### 4.5 Tokenization and delimiter safety

Adopt Kev's approach wholesale, including the part the Gemini draft omits entirely.

* Reuse rarely-used existing special tokens (`<|fim_prefix|>`, `<|fim_middle|>`, `<|box_start|>`,
  `<|box_end|>`, `<|fim_suffix|>`) as state, question, option-open, option-close, and decide
  delimiters. No embedding rows are added, so no new rows need training.
* Sanitize every caller-supplied string before tokenizing by rewriting `<|name|>` to `<¦name¦>`,
  so untrusted state text and option descriptions cannot forge a delimiter. The engine consumes
  attacker-influenced text (log dumps, customer messages, invoice fields). A forged `<decide>` token
  would let input text redirect the readout position. The Gemini draft has no coverage of this.
* Truncate state to 512 tokens (measured max is 491, so this truncates nothing today and leaves
  headroom) and fail loudly rather than silently truncating a branch.

## 5. Training

### 5.1 Data policy and split discipline

Build the whole pipeline on the 1,200-case train split first. That is the data Laya used to reach
its claimed 0.766, it is 6,000 labelled decisions, and it exactly matches the test distribution.

Partition the train split by case id, never by decision, so sibling questions stay together:

| Fold | Cases | Purpose |
|---|---|---|
| fit | 840 (70%) | Gradient updates for the backbone, pointer head, and ordinal term |
| calib | 180 (15%) | Temperature fitting, conformal quantile, head 3 training |
| dev | 180 (15%) | Early stopping, hyperparameter selection, ablation decisions |

Train head 3 and fit the conformal quantile with 5-fold cross-fitting inside `fit` plus `calib`, so
every correctness label comes from a model that did not see that case.

The test split is a vault. It is read exactly once per candidate model, by
`scripts/evaluate_verdict_baseline.py`, after all fitting is frozen. Anti-leak checklist in
section 8.

Do not generate 50,000 synthetic records before establishing whether real data suffices. The Gemini
draft budgets 5 to 6 hours on generation and decontamination for a 20-schema closed-set task with
1,200 labelled training cases already in hand. Synthetic data is a section 10 Phase 4 option with a
kill criterion, not a prerequisite.

### 5.2 Objective

Route by type, and train against the soft targets the harness scores against.

For choice and noul (K in {2, 4, 5}):

```
L_choice = KL( g || p_hat )  +  lambda_b * sum_k (p_hat_k - g_k)^2  +  lambda_h * CE(p_hat, label)
```

with `lambda_b = 0.5` and `lambda_h = 0.25` as starting values, swept on `dev` over
`lambda_b in {0.0, 0.25, 0.5, 1.0}` and `lambda_h in {0.0, 0.25, 0.5}`.

Rationale for including a Brier term, against the Gemini draft's explicit rejection of it: Brier is
one of the six metrics this benchmark reports, it is a strictly proper scoring rule, and Laya
already trains a composite of log score, spherical score, and RPS successfully
(`Laya/github-laya/laya/common.py::proper_reward`). The "competing gradient vectors" objection is
asserted in that draft without evidence. Treat it as a hypothesis and settle it with the lambda
sweep, which costs one afternoon.

Rationale for the small hard-label term: 1.55% of test items have `label != argmax(probabilities)`,
so pure distribution matching caps accuracy at 98.45%. The hard term recovers that.

For score:

```
L_score = KL( g || p_hat ) + lambda_r * RPS(p_hat, g) + lambda_h * CE(p_hat, label)
```

with `lambda_r = 1.0`.

Do not add a label-smoothing epsilon. The targets are already soft, with mean top mass 0.659.
Smoothing soft targets further flattens them below the panel and costs Brier.

### 5.3 Augmentation

Three augmentations, all free and all aimed at properties the harness or the product needs:

* Option order permutation on every choice question with K >= 3, resampled each epoch. If the
  masking in section 4.2 is enabled this is a no-op by construction, which makes it a live assertion
  that the mask works. If masking is disabled, this plus a symmetric-KL term is the fallback.
* State field dropout: drop one random top-level JSON key from the state with probability 0.15, and
  keep the original label. This teaches the model to spread probability mass when evidence thins,
  which is what the soft targets encode.
* Criteria paraphrase: the 20 schemas ship fixed criteria strings. Generate 5 paraphrases per option
  description once, offline, and sample among them. This is the only place a generative model is
  needed, it costs minutes, and it attacks the real overfitting risk (memorizing 20 fixed strings).

### 5.4 Schedule and hardware

The training set is 840 cases times 5 questions, which is 4,200 decisions per epoch packed into 840
sequences of about 516 tokens. This is a small job. Concretely:

* Backbone: `Qwen/Qwen2.5-0.5B`, loaded as `AutoModelForCausalLM(...).model`, vocabulary head
  discarded. Note it is not in the local Hugging Face cache yet, so budget a 1 GB download.
* LoRA r = 16, alpha = 32, dropout 0.05, on `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`,
  `task_type = FEATURE_EXTRACTION`. `peft` is not installed in `.venv`, so add it.
* Pointer head, ordinal term, and head 3 train in full precision, always.
* Optimizer AdamW, lr 2e-4 for LoRA and 1e-3 for the heads, weight decay 0.01, OneCycle with 10%
  warmup, gradient clipping at 1.0.
* Batch: one packed record per forward, gradient accumulation 8, 6 to 10 epochs with early stopping
  on `dev` accuracy plus Brier.
* Attention: standard eager or SDPA with the additive float mask from section 4.1. At L = 799 the
  full score matrix is 14 heads x 799 x 799 x 4 bytes, about 35 MB per layer transiently, which is
  not a problem on any of the target machines.

Do not use FlexAttention. Verified on this machine, torch 2.14.0:

```
torch.compile(flex_attention) on mps: NotImplementedError: FlexAttention does not support backward on MPS
torch.compile(flex_attention) on cpu: NotImplementedError: FlexAttention does not support backward on CPU
```

FlexAttention backward is CUDA-only. The Gemini draft makes it the mandatory training mechanism in
section 1.3 while offering "Training time on Apple Silicon MPS (M-series): 14 hours" in its
explanation document. That path does not exist. Uncompiled FlexAttention does run on MPS forward,
but it warns that it "materializes the full scores matrix instead of generating a fused kernel",
which removes the memory benefit that was the stated reason to use it.

Expected wall clock: 25 to 45 minutes per full run on a single T4 or A100, 2 to 4 hours on M-series
MPS. The Gemini draft's "8 hours 40 minutes on two T4 GPUs" is a budget for 52,000 records that this
plan does not generate.

## 6. Calibration and conformal prediction

### 6.1 Temperature

Fit temperature on the `calib` fold, grouped by `(qtype, K)`, which gives 5 groups here
(choice-4, choice-5, noul-2, score-4, score-5). Minimize negative log likelihood of the soft target,
and record the fitted values.

Do not impose the Gemini draft's `T >= 0.70` floor. Its justification, "prevents the optimization
from collapsing into Laya's pseudo-one-hot argmax exploit (T = 0.1006)", is not supported by
anything in the vendored Laya tree: Laya's shipped `eval/results.json` records fitted temperatures
of 1.637, 1.251, and 1.983, all above 1, which smooth rather than sharpen. Given section 2, a floor
at 0.70 would also block the region where ECE on the distribution is lowest. Let T range over
[0.25, 3.0], fit it, and print it.

Skip the parametric `T(K) = alpha * ln(K) + beta` form. With three distinct K values there are not
enough points to justify a 2-parameter curve over 5 independent scalars, and the "calibration
cliffs at bucket borders" the draft worries about cannot occur when the buckets are the entire
support of K.

### 6.2 Conformal sets

Implement split conformal with the adaptive prediction set (APS) score, calibrated on `calib`:

1. Score each calibration decision as the cumulative sorted probability up to and including the
   true class.
2. Take `q_hat` as the `ceil((n+1)(1-alpha))/n` empirical quantile.
3. At test time include sorted classes until the cumulative mass reaches `q_hat`.

Three corrections to the Gemini draft's section 3:

* APS sets are never empty. The top-1 class is always included, because the cumulative mass up to
  rank 1 is `p_(1)` and the inclusion test at rank 1 is satisfied by construction. The draft's
  "Empty set: out-of-distribution input" branch and the `cat_set.is_empty` check in its SDK example
  are dead code that can never fire. Remove them. Route out-of-distribution detection through a
  separate signal (max logit or energy score, thresholded on `calib`).
* Split conformal guarantees marginal coverage over the whole test distribution. The draft uses it
  as if it were conditional, dispatching autonomously whenever `|C(x)| == 1`. Conditioning on
  singleton sets selects the easy cases, so the guarantee does not transfer to that slice. If an
  auto-dispatch guarantee is needed, use Mondrian conformal stratified by `(workflow, question id)`,
  which is cheap here because there are only 20 strata, and report per-stratum empirical coverage.
* State where the calibration data comes from. The draft says "a held-out calibration set of
  n = 2,000 examples", and the only 2,000 in this project is the test set. Calibration comes from
  the `calib` fold of the train split. Never from test.

Add RAPS regularization (a penalty on set size beyond rank k_reg) if mean set size exceeds 2.0 at
alpha = 0.05, and tune `k_reg` on `dev`.

## 7. Backbone decision (settled 2026-09-19)

Earlier drafts of this plan proposed a bake-off between a decoder (Qwen) and an encoder. That is
now settled in favour of the encoder family, and Qwen is off the critical path.

Primary: `answerdotai/ModernBERT-large` (395.9M). This is Laya's actual backbone, and Laya's
published card records 0.766 accuracy with it on this exact benchmark, so the target is a known
reachable number rather than an extrapolation.

Development vehicle: `answerdotai/ModernBERT-base` (149.6M). Same code, one flag. A 30 to 45 minute
loop while debugging beats a 2 hour loop. Switch to large only for final runs.

Dropped from the critical path: `Qwen/Qwen3.5-0.8B-Base` (873.4M). The reasons are not about VRAM,
which it fits:

* Capacity is not the bottleneck. TF-IDF with no semantic understanding reaches 0.6505 here.
  ModernBERT-large reaches 0.766. The teacher panel agrees with itself unanimously on 59.4% of test
  decisions. The headroom above 0.766 is label noise and calibration, so the extra 477M parameters
  buy world knowledge that this closed-set benchmark does not reward.
* It costs the WebGPU demo. An encoder exports cleanly to ONNX Runtime Web. A decoder carrying
  custom block masks does not, and the in-browser playground is the project's differentiator.
* It costs the latency claim. Roughly 5.9x the compute of the 151M baseline, against 2.7x for large.

One caveat to state plainly: using Laya's exact backbone means a win here is a win on method, the
dual-head calibration fix in section 4.4, and not on scale. That is the better claim anyway, since
calibration is the slot the community tracker lists as unclaimed.

Note against the published comparison table: the ModernBERT-base row in Laya's table reads 0.646,
which is the same neighbourhood as the TF-IDF floor. Base is a development vehicle, not the ship
target. If base plateaus below 0.72, that is expected, and it is not a reason to abandon the design.

### Memory arithmetic for the 6 GB training card

Full fine-tuning, GTX 1660 Ti with about 5.5 GB usable:

| Backbone | fp32 params + grads | AdamW states | Total | Verdict |
|---|---|---|---|---|
| ModernBERT-base | 1.2 GB | 1.2 GB fp32 | about 2.8 GB | fits with default flags |
| ModernBERT-large | 3.2 GB | 3.2 GB fp32 | about 6.8 GB | does not fit |
| ModernBERT-large | 3.2 GB | 0.8 GB 8-bit | about 4.5 GB | fits with `--optim8bit --grad_checkpoint` |

Turing is compute capability 7.5, so bf16 and FlashAttention-2 are both unavailable. Use `--fp16`
and PyTorch SDPA. Operational detail is in `RUNBOOK.md`.

## 8. Evaluation protocol and anti-leak checklist

Every candidate model is scored by `scripts/evaluate_verdict_baseline.py` with no modifications
other than adding `ece_confidence` alongside the existing `ece`. Three seeds, report median and
spread, as `scripts/run_baseline_repeats.py` already does.

Before any test-set read, confirm in writing:

- [ ] Temperature was fit on `calib`, not test.
- [ ] Conformal quantile was fit on `calib`, not test.
- [ ] Head 3 was trained on out-of-fold predictions, not test.
- [ ] Early stopping and every hyperparameter choice used `dev`, not test.
- [ ] Ablation winners (option masking, cumulative link, lambda sweep) were chosen on `dev`.
- [ ] Criteria paraphrases were generated from train-split strings only.
- [ ] If synthetic data exists, its generator was never shown a test case, and the contamination
      scan in section 10 Phase 4 passed.
- [ ] The number of test-set evaluations run so far is recorded in the run manifest, and is small.

That last item is the one that actually protects the result. A benchmark read 40 times during
development is a validation set with extra steps.

Report these metrics together, always: accuracy, soft accuracy, Brier, `ece_distribution`,
`ece_confidence`, score MAE, within-one-level, conformal coverage at 0.90/0.95/0.99, mean conformal
set size, permutation argmax flip rate, latency p50 and p95, and the per-workflow accuracy
breakdown. Publish the reference rows from section 1.4 in the same table so readers can see the
floor.

## 9. SDK

Keep the existing `core/primitives.py`. It already has frozen Pydantic v2 `Choice`, `Score`, `Noul`,
their result types, and `INSUFFICIENT_EVIDENCE_ID`, with validators. It is better than the
replacement proposed in the Gemini draft section 5, whose flagship example does not run:

```
>>> Choice = Annotated[Enum, ChoiceMeta()]
>>> class IncidentTriageSchema(BaseModel):
...     category: Choice[IncidentCategory] = Field(description="...")
TypeError: typing.Annotated[enum.Enum, <ChoiceMeta object>] is not a generic class
```

`Annotated[Enum, ...]` is not subscriptable, so `Choice[IncidentCategory]` raises at class-definition
time. Separately, `severity: Score(1, 5)` puts a call expression in an annotation position, which
Mypy and Pyright both reject, so the claim that this design guarantees strict-mode compatibility is
inverted. If a decorated-enum ergonomic is wanted, the working form is a generic wrapper:

```python
from typing import Annotated, Generic, TypeVar
from enum import Enum
E = TypeVar("E", bound=Enum)

class ChoiceField(Generic[E]):
    """Marker carried in Annotated metadata; the engine reads it off the field."""
    def __init__(self, allow_abstention: bool = True) -> None:
        self.allow_abstention = allow_abstention

Category = Annotated[IncidentCategory, ChoiceField[IncidentCategory](allow_abstention=True)]

class IncidentTriage(BaseModel):
    category: Category
    severity: Annotated[int, ScoreField(min_score=1, max_score=5)]
```

Additions to the result object, on top of what `core/primitives.py` already returns:

* `confidence`: head 3's `q_hat`, documented as the probability that this answer matches the
  reference label.
* `distribution`: head 1's `p_hat`, documented as the estimated reviewer-panel distribution.
* `conformal_set`: options, `alpha`, `is_singleton`, `stratum`, and the empirical coverage measured
  for that stratum. No `is_empty`.
* `expected_value` and `standard_deviation` for score, computed from `p_hat`.

Keep policy in application code, which both drafts agree on and which is right.

## 10. Phases, gates, and kill criteria

Phase 0, diagnose before building (2 to 3 hours). Determine whether Verdict 1.0's 26.1% is a defect
or an architecture limit. Run three controls on the existing engine: score with option order
shuffled, score with option ids replaced by their descriptions, and score with the gold label
permuted. If shuffling options changes accuracy materially, or if accuracy rises above 0.269 under
any relabelling, the baseline has a wiring defect, and that defect must be understood before it is
carried into new code. Commit the section 1.4 reference rows as `reports/reference_floors.json`.
Gate: the diagnosis is written down. Nothing trains until it is.

Phase 1, harness and reference floors (2 hours). Add `ece_confidence` to the evaluation script, add
conformal coverage and set-size reporting, add the permutation flip-rate probe (copy Kev's
`permutation` and `isolation` measurements), and check in the TF-IDF baseline as a permanent
regression floor. Gate: `pytest` green, and the TF-IDF baseline reproduces 0.6505 +/- 0.005.

Phase 2, Track A and Track B bake-off (1 day). Both backbones, same folds, same objective, no
augmentation, no calibration. Gate: at least one track clears 0.700 accuracy on `dev`. If neither
does, stop and investigate the data path rather than scaling the model.

Phase 3, full training of the winner plus calibration (1 day). Lambda sweep, augmentation, option
masking ablation, cumulative link ablation, temperature fit, conformal calibration, head 3 training.
Gate: `dev` accuracy >= 0.760 and `dev` Brier <= 0.120 before the test vault is opened.

Phase 4, synthetic augmentation, optional (1 to 2 days, run only if Phase 3 lands between the floor
and the target). Generate on the order of 5,000 records across the 4 existing workflow families,
not 50,000 across 500 invented domains. Decontaminate with SHA-256 on canonicalized text plus
MinHash Jaccard at 0.5 plus embedding cosine at 0.84, which is the sensible core of the Gemini
draft's 5-tier gate. Kill criterion: if adding synthetic data does not improve `dev` accuracy by at
least 1.0 point, discard it and record the negative result.

Phase 5, freeze, single test read, report (half a day). Run the anti-leak checklist, evaluate three
seeds, write `reports/verdict_2_0_benchmark.json` with every metric from section 8 and every
reference row from section 1.4.

Total realistic budget: 3 to 4 working days, of which under 2 hours is GPU time for the primary
path. The Gemini draft's 15 to 19 hours is dominated by generating and decontaminating data this
plan does not need until Phase 4.

Things this plan does not do, deliberately:

* No 4-quadrant counterfactual abstention inoculation. The benchmark has no abstention option, and
  the premise is wrong on its own evidence: Kev's `runs/kev-vs-jev-v1.json` records mean
  `p_none` of 0.607 when "none of the above" is the correct answer and 0.205 when it is not, which
  is the correct direction. Jev is the one that under-uses abstention (0.457 versus 0.041). If an
  abstention primitive is wanted for the product, scope it as its own workstream with its own
  benchmark.
* No production logit adjustment for prior shift. It is a reasonable technique with nowhere to apply
  it on a benchmark whose test prior equals its train prior. Add it when a real deployment has a
  measured prior gap.
* No 500 enterprise domains or 12,000 schema topologies. If the goal is a generalization claim,
  build a held-out task-family benchmark and state the claim there, the way Laya's zero-shot table
  does (0.651 versus 0.753 in-task, which is where the interesting engineering is).

## 11. Reproducing every number in this document

```bash
# Dataset shape, cardinality, state lengths, teacher agreement
python scripts/analysis/profile_typed_decisions.py

# Reference floors: uniform, majority, TF-IDF, oracle
python scripts/analysis/reference_floors.py --out reports/reference_floors.json

# Brier and ECE feasibility frontier
python scripts/analysis/metric_frontier.py --out reports/metric_frontier.json

# FlexAttention backward support on this machine
python -c "import torch;from torch.nn.attention.flex_attention import flex_attention,create_block_mask;\
fa=torch.compile(flex_attention);bm=create_block_mask(lambda b,h,q,k:q>=k,None,None,256,256,device='mps');\
q=torch.randn(1,2,256,64,device='mps',requires_grad=True);fa(q,q,q,block_mask=bm).sum().backward()"

# Gemini SDK example
python -c "from enum import Enum;from typing import Annotated;from pydantic import BaseModel;\
C=Annotated[Enum,object()];\
exec('class S(BaseModel):\n    x: C[Enum]')"
```

The three analysis scripts do not exist yet. Writing them is the first task of Phase 1, and their
outputs replace every quoted number in this document with a committed artifact.
