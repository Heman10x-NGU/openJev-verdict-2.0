# Verdict 2.0 in plain English (Claude revision)

This document explains what we are building, what we found when we checked the assumptions behind
the Gemini draft, how long the work takes, and what could still go wrong. Everything numeric here
was measured on this machine. Section 11 of `IMPLEMENTATION_PLAN.md` lists the commands.

## The short version

We are replacing Verdict 1.0 with a small decision model that reads a workflow state once and
answers all five typed questions about it in a single forward pass. That part of the Gemini plan is
right and we are keeping it.

What we are changing is the target, the training data, the loss, and the way confidence is reported.
The Gemini plan was written against a benchmark that is much larger and much harder than the one we
are actually scored on, and it sets a pair of numeric targets that no model can hit at the same time.

## What we found when we checked

### The benchmark is smaller than the plan assumes

We loaded the dataset and measured it. `LocalLLaMA/typed-decisions` has 1,200 training cases and 400
test cases. Every case has exactly five questions. Across the entire dataset there are only 20
distinct question schemas (four workflows times five questions each), and they are the same 20 in
both splits. The largest number of options any question offers is five.

The states are short. The median state is 237 tokens and the longest is 491.

The Gemini plan was designed for 4,096-token states, 12,000 distinct schemas, 500 business domains,
and option sets of up to 64 candidates. None of that is present. A lot of the plan's machinery
(a 48 MB key-value cache, a cardinality correction factor, a two-phase inference engine to avoid a
memory explosion) is solving problems that do not occur at this size. The real cached state is
about 4 MB, and everything (state plus all five questions) fits in roughly 516 tokens.

### Two of the target numbers cannot both be achieved

This is the finding that changes the project.

The benchmark scores two things that sound similar and are not. Brier measures how close your
probability numbers are to the reviewer panel's probability numbers. ECE measures whether your
stated confidence matches how often you are actually right.

Here is the problem. The reviewer panel is genuinely split on most questions: its average top
probability is only 0.659. So a model that copies the panel perfectly says "66% sure" while actually
being right 98% of the time. That model gets a perfect Brier of 0.000 and a terrible ECE of 0.326.
If you sharpen the model until it says "95% sure" and its ECE drops to 0.046, its Brier climbs to
0.147.

We traced the whole curve. At the Brier target of 0.105, the best possible ECE is about 0.091. At
the ECE target of 0.040, the best possible Brier is about 0.150. The targets in the brief
(Brier at or below 0.105 and ECE at or below 0.040) sit in a region no predictor can reach, not even
a perfect one.

Our fix is to stop reporting one number as if it answered both questions. The model will output two
things: a probability distribution that estimates what the reviewer panel would have said, and a
separate confidence score that estimates the chance this particular answer is right. The first is
scored by Brier. The second is scored by ECE. Both are honest, and both can hit their targets
because they are now measuring different things. Laya's model already carries a small head doing
something close to this, so we know the construction works in this family of models.

We will publish both ECE figures side by side, including the original one, so nobody can accuse us
of choosing a flattering definition.

### The competitor comparison table does not hold up

We traced every number in the brief back to its source in the `Laya/` and `Kev/` folders.

The Jev row (72.70% accuracy, 0.148 Brier, 0.144 ECE, 710 ms) is a hardcoded Python literal inside
one of Laya's notebooks. Nobody in this repository measured it.

The Laya accuracy of 76.6% comes from a print statement in the publish cell of Laya's fine-tuning
notebook, and it is for Laya after fine-tuning on this benchmark's 1,200 training cases. Laya's
shipped evaluation file is for a completely different benchmark (23,024 questions about sentiment,
emotion, moderation, and so on) where it scores 0.753 in-task and 0.651 on held-out task families.
The Brier of 0.119 attributed to Laya belongs to the ModernBERT row of that same hardcoded table.

Kev has never been evaluated on this benchmark at all. Its 78.1% is a macro average over its own
suite (AG News, Banking77, BoolQ, MNLI, SST-5, Yelp). Its measured Brier on that suite is 0.302, not
0.112.

The latency figures are the most consequential mix-up. The "38 ms" credited to Kev is actually
Laya's single-question timing. Kev's own README says it serves a six-question request in about
160 ms on an Apple M5, and its demo output shows 162 ms. So the sub-35 ms target in the brief traces
back to a number that belongs to a different model answering one fifth as many questions.

We are not saying the competitors are worse than claimed. We are saying we do not know, because
nobody has run them on the same harness. Until someone does, those rows should be labelled as vendor
claims on other benchmarks.

### Our own baseline is broken in a way worth understanding first

Verdict 1.0 scores 26.10% accuracy. Guessing uniformly at random on this benchmark scores 26.90%.
Always answering with the most common label for each question scores 48.35%.

So Verdict 1.0 is not a weak model. It is performing below random guessing, and its Brier of 0.585
is 2.4 times worse than random guessing. A model that had simply learned nothing would score better.
Being reliably worse than chance takes information, and it usually means something is wired
backwards: options misaligned with their identifiers, descriptions swapped for ids, or a label
mapping inverted somewhere between the engine and the scorer.

Before we build anything new, we spend two hours finding out which. If there is a defect in how
options are passed to the scorer, rewriting the model on top of the same plumbing will carry the
defect forward and we will spend a week confused.

### A thirty-second baseline already beats most of the targets

We ran TF-IDF character n-grams into a logistic regression, one small model per question schema,
trained on the 1,200-case training split. It takes about thirty seconds on a laptop CPU with no GPU.

It scores 65.05% accuracy, Brier 0.145, ECE 0.031.

That baseline already beats the ECE target in the brief, comes within 0.04 of the Brier target, and
lands 7.7 points below the number attributed to Jev. This reframes the whole project. Any plan that
costs a day of GPU time has to beat 65% by a clear margin to have been worth running, and we now
have a permanent regression floor to check ourselves against.

### The training path in the Gemini plan does not run

The Gemini plan requires PyTorch FlexAttention for training, and offers Apple Silicon as a supported
target with a 14-hour estimate. We tested it on this machine with PyTorch 2.14:

```
FlexAttention does not support backward on MPS
FlexAttention does not support backward on CPU
```

The backward pass is CUDA only. Training on the Mac would fail on the first gradient step. Even the
forward pass on MPS warns that it materializes the full attention matrix rather than using a fused
kernel, which removes the memory saving that was the reason to use it.

The replacement is simple and already proven: Kev passes an ordinary additive attention mask to
standard PyTorch attention, which backpropagates fine everywhere. At our sequence lengths the
attention matrix is about 35 MB per layer, which is not a problem on any machine we own.

### The headline SDK example raises an error

We ran the developer-facing code from section 5.2 of the Gemini plan verbatim:

```
TypeError: typing.Annotated[enum.Enum, <ChoiceMeta object>] is not a generic class
```

`Annotated[Enum, ...]` cannot be subscripted, so `Choice[IncidentCategory]` fails when the class is
defined. Separately, writing `severity: Score(1, 5)` puts a function call where a type belongs,
which both Mypy and Pyright reject, so the plan's claim of strict-mode compatibility is backwards.
We already have a working, validated Pydantic v2 layer in `core/primitives.py`, and we keep it.

### Two more things we corrected

The Gemini plan says Verdict 1.0 fails because ModernBERT uses sliding-window attention on 18 of its
28 layers. ModernBERT-base has 22 layers, and its config sets global attention every third layer, so
eight layers see everything. The plan's own architecture section says 22, contradicting its
explanation. The sliding window is not the cause of the failure.

The plan also devotes a whole training-data strategy to inoculating against a Kev bug where Kev
supposedly picks "none of the above" whenever that phrase appears. Kev's own published measurements
say the opposite: Kev gives "none" an average probability of 0.607 when it is the right answer and
0.205 when it is not, which is the correct direction. Our benchmark has no abstention option at all,
so the whole workstream is aimed at a problem that is neither real nor ours.

## What we are keeping from the Gemini plan

Four ideas are good and survive intact.

Isolating each question into its own branch, with a shared state prefix and restarted position
numbering, is correct. It is also exactly what Kev already does, and Kev measured that packing five
questions into one sequence gives the same answers as running them separately (agreement to six
decimal places) while running twice as fast.

Masking candidate options so they cannot see each other is a genuine improvement. Kev flips its
top answer 7.4% of the time when you reorder the options, because in a normal causal model later
options can read earlier ones. Masking makes the answer provably independent of option order. We
will still run the ablation, because removing the ability to compare candidates against each other
could cost accuracy, and we would rather have the number than the principle.

Treating ordinal scores differently from unordered choices is right. We use a ranked probability
term rather than the plan's ordered logistic model, because an ordered logistic model with one
latent value can only produce single-peaked distributions, and 3.8% of the score targets in this
benchmark have two peaks. We will run the ordered logistic version as a recorded comparison.

Conformal prediction sets are worth shipping, with three corrections. The sets can never be empty,
so the plan's "empty set means out of distribution" rule is code that can never run. The coverage
guarantee is an average over all decisions, not a promise about the specific decisions you choose to
automate, so we stratify by question schema (there are only 20) to get a guarantee that actually
applies. And the calibration data has to come from the training split, because the only 2,000-item
set in this project is the test set, and calibrating on it would invalidate the result.

## How long it takes

The Gemini plan budgets 15 to 19 hours, most of it generating and cleaning 50,000 synthetic records.
We do not need those records to start, because we already have 6,000 real labelled decisions that
match the test distribution exactly. Here is the revised budget.

### Phase 0: diagnose the current baseline (2 to 3 hours)

Find out why Verdict 1.0 scores below random. Run the engine with options shuffled, with ids and
descriptions swapped, and with labels permuted. Write down what we learn. Nothing else starts until
this is understood.

### Phase 1: harness and floors (2 hours)

Add the second ECE figure, conformal coverage, set size, and the permutation flip-rate probe to the
evaluation script. Check in the TF-IDF baseline as a permanent regression floor. Commit the
reference numbers (random, majority, TF-IDF, oracle) as a JSON artifact so every future report can
print them alongside the model.

### Phase 2: two-model bake-off (1 day)

Train two versions on identical data with identical settings. One uses Qwen2.5-0.5B, the decoder
that Kev uses. One uses ModernBERT-base at 149M, the encoder family that Laya uses, which is 3.4
times smaller. For a task with 20 fixed label sets and short inputs, the small encoder may well win,
and it is the only realistic route to a fast laptop number. We pick on measured accuracy at a
latency budget rather than on which model sounds more impressive.

### Phase 3: full training and calibration (1 day)

Train the winner properly. Sweep the loss weights, run the option-masking and ordered-logistic
ablations, add the three augmentations, fit temperature per question type, fit the conformal
threshold, and train the confidence head on held-out folds.

The training run itself is short. With 840 training cases and a LoRA adapter, expect 25 to 45
minutes on a single T4 or A100, or 2 to 4 hours on an Apple Silicon Mac. The Gemini plan's
8 hours 40 minutes was a budget for 52,000 records.

Note one practical item: Qwen2.5-0.5B is not in the local model cache, so budget a 1 GB download,
and `peft` is not installed in the virtual environment yet.

### Phase 4: synthetic data, only if needed (1 to 2 days, optional)

Run this only if Phase 3 lands between the floor and the target. Generate about 5,000 records across
the four workflow families we already have, not 50,000 across 500 invented domains. Screen them for
contamination with exact hashing, MinHash overlap, and embedding similarity, which is the sensible
core of the Gemini plan's five-tier gate. If the synthetic data does not add at least one accuracy
point on the dev fold, throw it away and write down the negative result.

### Phase 5: freeze and report (half a day)

Walk the anti-leak checklist, evaluate three seeds, publish every metric together with the reference
floors.

Total: 3 to 4 working days, of which under two hours is GPU time on the main path.

## The targets we are committing to

| Metric | Floor that blocks shipping | Target | Stretch |
|---|---|---|---|
| Accuracy | 0.700 | 0.790 | 0.815 |
| Brier on the reported distribution | 0.140 | 0.105 | 0.090 |
| ECE on the confidence channel | 0.060 | 0.040 | 0.030 |
| Conformal coverage at 95% | 0.93 | 0.95 | 0.95 with sets under 1.8 options |
| Score MAE | 0.500 | 0.420 | 0.380 |
| Latency, 5 questions, M-series Max | 200 ms | 120 ms | 90 ms |
| Latency, 5 questions, T4 | 80 ms | 45 ms | 25 ms with INT8 |

The floor is set five points above the TF-IDF baseline. If the trained model cannot clear it, the
right answer is to ship the classical baseline and say so publicly, which would still be a better
result than what Verdict 1.0 does today.

The 81.5% stretch number is kept because it is worth aiming at, and the evidence for reaching it is
Laya's claimed 0.766 after fine-tuning on the same training split. But that claim is unverified, and
we should say out loud that 0.815 is a hope rather than a forecast until somebody measures a
competitor on this harness.

## What could still go wrong

The reviewer panel disagrees with itself on roughly 40% of test decisions, so a large slice of the
remaining error may be irreducible. We will know which slice by breaking accuracy out by panel
agreement level, and we should do that before concluding the model has a ceiling.

Masking options from each other guarantees order invariance but removes the model's ability to weigh
candidates against each other in one pass. That could cost accuracy. The ablation in Phase 3 settles
it, and we take the number over the principle.

Reporting confidence from a separate head is the right engineering answer to a real metric conflict,
and it will look like metric gaming to a skeptical reader. The defense is to publish both figures,
explain the conflict with the frontier table, and let people check the arithmetic.

Finally, the benchmark is small: 400 test cases and 2,000 decisions. The standard error on an
accuracy near 0.80 is roughly 0.9 points, so differences under about two points are noise. We should
report confidence intervals and resist celebrating a one-point win.
