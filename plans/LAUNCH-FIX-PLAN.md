# Launch fix plan v2

Supersedes v1. v1 treated this as a receipts-hygiene problem. After reading the
three competing repositories in `oss/` and the live X discussion, the positioning
is also wrong, and there is a confound inside the benchmark itself.

---

## Part A: What the discussion is actually about

Every serious post in the thread converges on one idea: a decision is a
constrained readout over a fixed option set in one forward pass, instead of
generated text that code parses back into an enum. The disagreements are narrow
and specific.

1. Anders Lie argues the speed comes from the inference pattern, not the
   architecture: prefill the shared state once, fork the KV cache per question,
   read logits restricted to the valid choice labels. Any open-weight LLM can do
   this. That post has 43K views and is the consensus explanation.
2. Jon Magoon argues the softmax over a constrained set is a conditional
   distribution over those options, not confidence. Nobody in the thread has
   answered this with data.
3. 2FreeTheMoney asks whether the probabilities survive reordering the choices.
   Unanswered.
4. Regev Bensimon asks how multi-token labels are handled, because joint
   probability is affected by label length. Unanswered.
5. Sean Brynjolfsson draws the real line: Jev is positioned to reason about the
   world and take state-changing actions. GLiNER-class models never promised to
   add external information.
6. Joshua K tells the GLiNER co-founder to stop comparing fine-tunes to Jev.
   That is the reception any encoder-based comparison gets by default.

Points 2, 3, and 4 are the open questions. They are also the only questions this
repository is equipped to answer.

## Part B: What the three repositories in `oss/` already do

### `oss/open-jev` (Joshua Penman, DiffusionGemma 26B-A4B)

Block-diffusion model generating a whole JSON canvas at once. `constraints.py`
masks each canvas position to a set of allowed label tokens and forces EOS after.
`label_tokens()` verifies each label encodes as exactly one token and rejects the
tokenizer otherwise, which sidesteps the label-length problem by construction.
It records the pre-temperature choice logits so probabilities are recoverable.

Results: 88.4 percent on 337 scored questions from TypeSafe's public evaluation
cases against Jev's saved 90.8 percent, 408 of 408 valid typed answers, 4.85x
faster and 79 percent cheaper grouped versus one question per sequence.

It discloses that questions on one canvas attend to each other, so it does not
reproduce Jev's question isolation, and that the zero hallucination rate is a
harness guarantee rather than a model property.

### `oss/open-jev-v2/jelike/jevlike` (from-scratch or frozen encoder plus scorer)

Each option becomes a query vector that attends over the context, a shared dot
product scores each option and context pair, and a softmax runs across options.
Rows may carry different option counts. Trains from scratch on byte embeddings or
on a frozen HuggingFace encoder.

Two things it ships that this repository does not:
- A **shuffled-context control** printed by the eval command, pairing each option
  menu with the wrong context. A useful model must beat it.
- Split guidance that keeps related records together to stop near-duplicate
  leakage.

Its honesty is the reason to study it: 26 percent on target-disjoint Wikispeedia
against 8 percent for controls, and a chess checkpoint reported as 0 wins,
2 draws, 48 losses against Stockfish level 0.

### `oss/open-jev-v2/openjev` (Qwen3.5-4B frozen, direct logit readout)

This one owns `openjev.com`, ships a WebGPU browser demo, and leads with
"Wow! No waitlist. Run it in your browser today." Runtime-defined criteria and
option descriptions, shared-state prefill with parallel suffixes, committed raw
timings, prompt hashes, and model revisions.

Results: 0.845 modal agreement on the TypeSafe selected subset against published
Jev 0.883, 0.637 balanced accuracy on WANLI, 1.023 s for 21 probability pairs
versus 5.332 s for the compact generative baseline.

It ships the robustness matrix this repository lacks:

| Variant | Direct accuracy | Direct flips |
| :--- | ---: | ---: |
| Option reversal | 0.813 | 10 of 36 |
| Criterion wrapper | 0.706 | 9 of 36 |
| Irrelevant context | 0.821 | 4 of 36 |

And it states plainly: "The scores therefore cannot be treated as Jev-like
operational calibration."

## Part C: Consequences

1. **The name is gone.** Two of the three repositories are called OpenJev and one
   holds the dot-com with a live browser demo. Rename.
2. **The browser demo is not a differentiator.** `openjev.com` shipped it with
   almost the exact hook in the current draft tweet.
3. **Banking77 is invisible to this audience.** Every competitor reports against
   TypeSafe's public evaluation cases and Every's judgment lab. A Banking77
   number cannot be compared to anything anyone in the thread is discussing.
4. **The open gap is calibration and robustness.** Every competitor explicitly
   disclaims it. This repository has a trained checkpoint, held-out splits,
   temperature scaling, ECE, Brier, NLL, a selective-risk curve, and a failure
   gallery. That is the position: not another replica, the audit none of the
   replicas ran.

## Part D: The confound inside the benchmark

`core/banking_glossary.py` contains 53 curated descriptions. Banking77 has 77
categories. The test set uses all 77. `get_enriched_label()` falls back to a
template for the other 27:

```
curated:  "Inquire about whether a newly ordered debit or credit card has
           arrived or when it will arrive in the mail"     (20 words)
fallback: "Inquiry or request regarding Refund not showing up"   (8 words)
```

So the benchmark contains two classes of label with systematically different
length and semantic richness, and which class the gold option falls into is not
controlled. `WALKTHROUGH.md` claims the glossary "replaced shorthand tags with
natural language definitions", which holds for 65 percent of categories.

This is Regev Bensimon's question landing on this repository, and it is a five
line script for anyone to run: split accuracy by whether the gold label was
curated or templated. Do it first.

---

# The plan

Phase order is mandatory. Phases 1 to 4 produce artifacts. Phase 5 generates all
documentation from those artifacts.

## Phase 0: Lock decisions

**0.1 Name.** Replace `OpenJev` across 26 files. It must not contain "jev".
Candidates: `Verdict`, `Quorum`, `Readout`.

**0.2 Positioning.** One sentence, used everywhere: a calibration and robustness
audit of single-pass decision readout, with a small open encoder as the subject.
Not a Jev replica. Not a Jev comparison.

**0.3 Browser model variant.** fp16 for the browser, fp32 as the Python
reference. Publish a quality parity table between them.

## Phase 1: Blockers

### 1.1 Publish the weights
`.gitignore` excludes `*.onnx`, `*.safetensors`, `*.pt`, so nobody who clones can
run anything. The ONNX file is 606,323,181 bytes, which GitHub rejects, and LFS
free tier allows roughly one and a half downloads per month.

Upload to HuggingFace. Write `scripts/download_artifacts.py` reading
`artifacts/ARTIFACTS.json` (repo id, filename, sha256, byte size) and verifying
every hash, exiting non-zero on mismatch. Write a model card carrying the
limitations section.

Acceptance: fresh clone, `python scripts/download_artifacts.py`, then
`CI_REQUIRE_ARTIFACTS=1 pytest tests/ -q` passes with no skips.

### 1.2 One prompt contract for Python and the browser
`webgpu-demo/worker.js:113` builds `<<LABEL>>desc...<<SEP>>{text}`. Training and
evaluation build `<<LABEL>>desc...<<SEP>>Question: {q}\n\nContext:\n{text}`
(`scripts/train.py:222` into `core/formatting.py:41`). The demo also uses
shortened candidate descriptions that differ from the glossary.

Add `core/formatting.py::build_model_input(question, context, label_descriptions)`.
Refactor `train.py`, `evaluate.py`, and `core/engine_encoder.py` to call it. No
caller may concatenate a prompt itself. Emit `webgpu-demo/prompt_contract.json`
from the Python constants and have `worker.js` build from that template. Generate
`webgpu-demo/presets.json` from the glossary rather than hand-writing it.

Add `tests/test_browser_parity.py` asserting the Python string and the contract
applied to the same fixture are byte-identical.

Acceptance: the browser and `scripts/evaluate.py` agree to within 1e-3 on the
same record.

### 1.3 Worker bugs
All in `webgpu-demo/worker.js` unless noted.

1. Session leak: `initEngine` reassigns `session` without `release()`
   (`worker.js:45`, `:55`). The checkpoint dropdown at `index.html:676` makes
   this reachable in one click.
2. Init race: no guard on concurrent `INIT`. Hold an `initPromise`.
3. Inference race: `self.onmessage` does not await the previous `session.run`.
   Serialise behind a promise chain.
4. False backend label: `["webgpu", "wasm"]` lets ORT fall back silently while
   `usedProvider` is set to `"WebGPU"` unconditionally (`worker.js:49`). Try
   `["webgpu"]` alone, catch, then `["wasm"]` in a separate call.
5. Stale temperature: `calibratorTemp = 3.224806` (`worker.js:19`) is the v1
   value used whenever the calibrator fetch fails. Remove the fallback and error.
6. Hardcoded size: `modelSizeMb = 578.2` (`worker.js:20`). Report real bytes.
7. Replace `InferenceSession.create(url)` with a helper that checks the Cache
   API, streams the body through a reader posting byte progress, caches it, and
   calls `create(arrayBuffer)`. This fixes the silent multi-minute load, the
   repeat 600 MB download per visit, and gives a real size.
8. Tokenizer bounds: pass `{ truncation: true, max_length: 1024 }` and record
   truncation in the receipt.
9. XSS: `cand.id` and `cand.desc` go through `innerHTML` in `renderCandidates`.
   Use `textContent`.
10. Receipt: rename `calibrated_probabilities` to `probabilities` and add
    `temperature_applied`, since the field currently claims calibration while
    `calibration.status` says `unvalidated_scope`.

### 1.4 No hardcoded metrics in the demo
`index.html:363` hardcodes `ECE = 1.59% (Calibrated)`, a v1 number, while the
default checkpoint is v2 at 4.70 percent. Render every badge and the whole audit
table from the evaluation JSON at runtime.

Acceptance: grep for `1.59`, `93.5`, `93.7`, `70.5`, `95.5`, `21.4` in
`webgpu-demo/index.html` returns nothing.

## Phase 2: Correct the record

### 2.1 Select on a proper scoring rule
`scripts/train.py:304` selects on `val_accuracy >= best_accuracy`. Validation NLL
went 0.283, 0.332, 0.343 and Brier went 0.1387, 0.1413, 0.1448 across three
epochs while accuracy moved 0.912 to 0.916, which is two examples out of 500. The
saved checkpoint is the worst-calibrated one measured.

Add `--selection_metric` defaulting to `val_nll`. Save a checkpoint per epoch
under `artifacts/v2/epoch_{n}/`. Record the criterion, the selected epoch, and
the full per-epoch table in the manifest. Add linear warmup plus cosine decay,
which is absent and is part of why later epochs degrade. Retrain, which took 891
seconds on MPS. If the NLL-selected checkpoint has worse abstention recall,
publish both and state the tradeoff.

### 2.2 Report calibration in both directions
`WALKTHROUGH.md` section 2 item 4 says temperature scaling reduced calibration
error to 4.70 percent on v2. The receipt shows uncalibrated ECE 3.96 percent and
calibrated 4.70 percent, equal-mass 3.96 to 4.42. NLL and Brier improved, as they
must when T is fitted on NLL.

Publish uncalibrated and calibrated columns side by side everywhere. Add
`min_bin_count` to `compute_ece` defaulting to 10 and exclude smaller bins from
MCE, because the reported MCE of 0.405 comes from a bin holding three samples.
Emit bin counts in the rendered docs, distinguishing empty bins from
zero-accuracy bins, which both currently serialise as `0.0`. Add adaptive-binning
ECE and a 1000-resample bootstrap CI for ECE, accuracy, and both abstention
rates. With 927 of 1000 predictions in the top bin, a point estimate carries
almost no information.

### 2.3 Precision beside recall, always
v1 abstention precision was 97.74 percent, v2 is 88.57 percent, and the
regression appears in no table and no draft. v2 bought 6.5 points of recall with
9 points of precision, roughly 24 false abstentions on in-scope queries. Every
abstention figure ships as recall, precision, F1, and false-abstention count.

### 2.4 Failure gallery
`reports/v2/failure_gallery.md` concludes an 85 percent threshold "safely gates
the majority of these errors" directly beneath failures at 88.6, 89.9, 90.5, and
93.4 percent. Generate it from `predictions_v2.jsonl`, add an above-threshold
column, sample across the confidence range, and replace the note with the
measured count.

### 2.5 Housekeeping
Parameter count is 151,378,177 in the manifest and `149M` in every receipt
string; derive it. `tests/test_export_parity.py:29` targets the v1 path, so the
shipped v2 ONNX has never been parity-checked; parameterise over every bundle and
fail rather than skip under `CI_REQUIRE_ARTIFACTS=1`.
`tests/test_engine_encoder.py` runs `DecisionEngine(model="mock")` while the docs
describe it as real single-pass evaluation; fix the description or add a real
test behind a marker.

## Phase 3: The experiments that answer the thread

Each writes a JSON receipt under `reports/v2/`. Publish every result, including
the bad ones. Several will be worse than the current headline. That is the point.

### E1: Shuffled-context control (from jevlike)
Pair each test record's option menu with a different record's context. Report
accuracy. There is currently no control baseline of any kind, so nothing
establishes that the model reads the input rather than exploiting label priors.
This is the cheapest and most damaging missing experiment.

### E2: Option-order sensitivity (answers 2FreeTheMoney)
For every test record, evaluate the original order, the reversed order, and three
seeded shuffles. Report argmax flip rate, mean total-variation distance between
probability vectors, and whether flips correlate with confidence. Separately
report whether P(abstain) depends on the abstention option's index, since it is
shuffled into a random position.

`oss/open-jev-v2/openjev` measured 10 flips of 36 on option reversal. A comparable
number here is expected and publishable.

### E3: Label-length and glossary-coverage bias (answers Regev Bensimon)
27 of 77 categories have no curated glossary entry and get a template fallback.
Report accuracy split by whether the gold label was curated or templated, the
correlation between label token length and selection probability, and a rerun
with all 77 categories curated to a uniform length band. Until this runs, part of
the 93.5 percent is measuring glossary coverage.

### E4: Hard-negative distractors
Distractors are sampled uniformly over 76 categories, so the confusable sibling
almost never appears. Embed all 77 descriptions with the backbone, build K=5 and
K=9 slices using nearest non-gold neighbours, and evaluate. This is why 93.5
percent at K=5 coexists with 74 percent at K=25, and why every failure in the
gallery is a near sibling.

### E5: Abstention generalisation
The missing-option training samples and the evaluation slice come from the same
function, same K, same distractor distribution, same abstention string, so the
70.5 percent is in-distribution recall on a train-time augmentation. Build a
missing-option slice at K=9 with hard negatives, and one with the abstention
description reworded to a held-out synonym. Report both beside 70.5 percent.

### E6: Contamination audit
Decontamination is lowercase exact string match
(`scripts/download_real_benchmarks.py:159-164`), and Banking77 is dense with
paraphrases. Compute character 5-gram Jaccard across every train-test pair,
report counts at 0.7, 0.8, 0.9, 1.0 plus the twenty highest pairs verbatim, rerun
the test set with pairs above 0.8 removed, and publish the delta. Replace the
word "decontaminated" with the method actually used.

### E7: Latency with percentiles
`21.46 ms` appears only in prose with no script, no distribution, no sample
count. `artifacts/v2/WALKTHROUGH.md` presents 23.58 to 21.46 ms as a v1 to v2
improvement, which is impossible on an identical graph. Write
`scripts/benchmark_latency.py`: 20 warmup, 200 timed, per K in {3,5,9,17,25},
reporting p50, p90, p99, mean, stddev, processor, ORT version, thread count, and
token count. Benchmark fp16 and the WASM single-thread path, which is what most
demo visitors experience. Delete the v1 to v2 delta claim.

### E8: fp16 parity
Evaluate fp16 on the full test set and every slice. Publish fp32 versus fp16 for
accuracy, NLL, Brier, ECE. If accuracy moves more than 0.5 points, ship fp32.

### E9: One legible external number
Every competitor reports against TypeSafe's public evaluation cases
(evals.typesafe.ai) or Every's judgment lab. A Banking77 number is not comparable
to anything in the discussion. Score the same public subset the other two repos
used, as a fourth column beside their published figures, using the same
constrained-readout interface. If the encoder does poorly, publish that. A
measured floor is a result.

This is the one item that can be cut if time is short, at the cost of the release
being uncomparable to its three peers.

## Phase 4: Rename
`sed` every `openjev` and `OpenJev` across the 26 files listed by
`grep -rl "openjev\|OpenJev" . --exclude-dir=node_modules --exclude-dir=.git --exclude-dir=.venv --exclude-dir=oss`.
Rename artifact files. Bump `receipt_version` to `2.0.0`. Leave `oss/` untouched.

## Phase 5: Generate the docs from the receipts
Four current blockers exist because numbers were typed into three places that
then drifted.

Write `scripts/render_receipts.py` rewriting all tables in `README.md` and
`WALKTHROUGH.md` between `<!-- BEGIN GENERATED: name -->` markers. Convert README
wholly to v2; it currently carries 93.70 percent, ECE 1.59 percent, missing-option
9 percent, 68.0 percent coverage, and a cardinality table that disagrees with the
v2 report at every K, so the front page argues against the launch post. Delete
`artifacts/v2/WALKTHROUGH.md` and `reports/v2/WALKTHROUGH.md`. Add
`tests/test_docs_fresh.py` failing when committed docs differ from regenerated
ones.

Rewrite Known Boundaries to lead with E1 through E6 and the precision regression.

## Phase 6: Verification gate
Run from a fresh worktree, not the working tree.

```
git worktree add /tmp/verify-launch HEAD
cd /tmp/verify-launch
python -m venv .venv && ./.venv/bin/pip install -e .
./.venv/bin/python scripts/download_artifacts.py
CI_REQUIRE_ARTIFACTS=1 ./.venv/bin/python -m pytest tests/ -q
./.venv/bin/python scripts/render_receipts.py --check
grep -rn "1\.59\|93\.70\|21\.46" README.md WALKTHROUGH.md webgpu-demo/index.html
```

The grep must return nothing. Then load the demo with WebGPU on and off, switch
checkpoints ten times, and confirm memory is stable and the badge is truthful.

## Phase 7: Launch copy
Write only after Phase 5 produces final numbers. Constraints: under 280
characters per post, no em dashes, precision always beside recall, no latency
without a percentile, and the architecture concession inside the post rather than
in a reply. The hook is calibration and robustness, because the browser demo hook
is taken.

---

## Out of scope
int8 quantisation, Score and Noul primitives, any merge to a default branch.
