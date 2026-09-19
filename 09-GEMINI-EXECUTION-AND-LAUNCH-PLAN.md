# RLCD: Execution, measurement, and launch plan for Gemini

Author: ChatGPT 6 Astra

Created: 2026-09-17T13:10:44+05:30 (Asia/Kolkata)

Reviewed commit: `fa70460a31c4716887c3b1d5bd1c1b73f367b8c1`

Status: Implementation instructions. The work below has not been implemented or released.

You should read this alongside `08-ENGINEERING-IMPLEMENTATION-PLAN-ASTRA.md`. Document 08 supplies the technical architecture and mathematical corrections. This document reviews the implementation that followed it and specifies the shortest credible release path. Preserve the engineering invariants in `AGENTS.md`.

## 1. The assessment

You have a real fine-tuned classifier, a real ONNX export, and useful SDK scaffolding. You do not yet have the browser decision engine described in the launch draft. The browser displays fixed predictions and simulated timings. Several SDK contracts also permit misleading confidence claims.

The 96% result is reproducible on the existing test. Its task is restricted candidate selection: each in-scope example includes the correct Banking77 intent, three random distractors, and abstention. It is not classification among all 77 intents. You should preserve this result with its exact task definition, then add harder evaluations.

The strongest next release is a small, inspectable decision SDK with a real browser playground and a reproducible abstention benchmark. Let developers enter text, modify the candidate set, remove the expected answer, and download the actual inference receipt. That gives them something useful to install and something concrete to challenge.

You cannot engineer a guaranteed number of GitHub stars. You can improve the reasons people try, trust, reuse, and share the project. Prioritize those outcomes over a larger collection of claims.

### What this review verified

The review covered the pasted execution log, documents 00 through 08, `AGENTS.md`, `README.md`, `WALKTHROUGH.md`, the untracked implementation plan, all first-party Python source and tests, the complete browser page, packaging, artifact metadata, and every local dataset record programmatically. The original architecture documents were checked against their previously read hashes. Two Luna Max reviewers inspected runtime contracts and the browser/export paths. Binary checkpoints were inspected through metadata and actual inference; bundled reference repositories and installed dependencies were not treated as first-party implementation.

The root independently ran the local test suite and loaded the upstream and fine-tuned models on CPU with network access disabled. The tests passed: 25 tests in 3.47 seconds, with one Torch deprecation warning. Passing these tests does not validate the simulated browser page or general calibration claims.

| Evidence | Result | Interpretation |
| --- | --- | --- |
| Fine-tuned checkpoint | 151,378,177 stored FP32 parameters, 605,529,340 bytes | A real approximately 151M-parameter GLiClass checkpoint, including an extra scalar in this installed implementation |
| ONNX graph | 606,323,181 bytes, FP32, opset 17, output `[batch, 25]` | A real local graph with a 25-candidate output capacity |
| Original 1,000-case test, upstream weights | Accuracy 53.5%, summed Brier 0.653389, equal-width ECE 0.227020 | Reproduced for the restricted five-option task |
| Same test, fine-tuned weights and saved temperature | Accuracy 96.0%, summed Brier 0.071856, equal-width ECE 0.020107, NLL 0.147008 | Reproduced; the walkthrough's NLL values need correction |
| Saved fine-tuned results before temperature scaling | Brier 0.070446, equal-width ECE 0.024145, equal-mass ECE 0.020003 | Fine-tuning accounts for most of the reported improvement |
| Saved fine-tuned results after temperature scaling | Brier 0.071856, equal-width ECE 0.020107, equal-mass ECE 0.022104 | Scaling improves NLL and one ECE estimator, while Brier and the other ECE estimator worsen slightly |
| Targeted diagnostic: one in-scope example per intent, original five options | 73/77 correct, 94.81% | A small diagnostic subset, not a separate benchmark |
| Same 77 examples with the labeled correct option removed | 16/77 abstentions, 20.78%; 61 alternative answers, including 22 at confidence at least 0.90 | The model's strong rejection of distant CLINC examples does not establish reliable rejection when a plausible banking answer is missing |
| Attempted 77 intents plus abstention | Evaluation raises `IndexError: Target 76 is out of bounds` | The current model cannot be advertised as supporting this candidate count |
| Browser inference and JSON comparison | Timers and hardcoded probabilities/strings | No browser inference latency or browser calibration was measured |

The missing-option diagnostic changes the existing task after training and uses an explicit expected-abstention target. It exposes a failure mode. It is not an estimate of production error or a fresh statistical test set. Closely overlapping candidate descriptions still need adjudication in a formal missing-option benchmark.

The persisted review receipt is `reports/astra-review-2026-09-17/review.json`. Its companion `predictions.jsonl` contains the raw logits and probabilities for 2,154 evaluated rows across the four completed experiments. Accuracy, Brier, and NLL were independently recomputed from those saved rows. The failed 78-way attempt is recorded as an unsupported-capacity result, not a completed benchmark.

### Release blockers and their locations

| Priority | Finding | Location at the reviewed commit | Required correction |
| --- | --- | --- | --- |
| P0 | Browser results are fabricated presentation fixtures | `webgpu-demo/index.html:361-396`, `459-518` | Replace them with model execution and measured timing; remove the simulated comparison |
| P1 | Accepted schemas can exceed the model's capacity and lose candidates | `core/primitives.py:53`, `core/engine_encoder.py:170-175` | Validate capacity before tokenization and assert one logit per candidate |
| P1 | Calibration can be attached to unrelated weights, schemas, or domains | `core/calibration.py:244-258`, `core/engine_encoder.py:187-214` | Bind calibration to an artifact and evaluated scope; expose unvalidated use |
| P1 | Calibration split has no out-of-scope records | `scripts/download_real_benchmarks.py:164-175` | Separate source pools before sampling; assert actual split composition |
| P1 | Injected model/tokenizer path crashes because `_pipe` remains `None` | `core/engine_encoder.py:64-67`, `163-168` | Use one explicit formatter and tensor preparation path |
| P1 | Numerical failures and invalid probabilities can enter result objects | `core/primitives.py:146-189`, `core/calibration.py:36-45` | Validate finite values, support masks, targets, and result invariants |
| P1 | Missing weights silently fall back to another model during export/evaluation | `export/export_onnx.py:46-58`, `scripts/evaluate.py:207-220` | Fail on missing requested artifacts; make baseline mode explicit |
| P1 | Public claims exceed the implemented scope | `WALKTHROUGH.md:3-57`, `80-124` | Generate claims from verified receipts and identify fixture-only workflows |
| P2 | `Noul` silently defaults to a new semantics version | `core/primitives.py:138-140` | Require the semantics field and reject omission |
| P2 | Custom table row-key name is lost during reconstruction | `core/tabular.py:62-70`, `111-128` | Preserve the original key and types in the table representation |
| P2 | Causal confidence omits full-vocabulary allowed-set mass | `core/engine_causal.py:243-255` | Report the conditional nature of slot probabilities and the missing diagnostic |
| P2 | Export parity assertions are weaker than the published statement | `tests/test_export_parity.py:49-91` | Publish measured errors and test capacity, batching, precision, and browser execution |

The example runner evaluates five fixtures and builds routing strings. It does not ingest a live webhook, issue refunds, or send Slack messages. Keep deterministic policy examples, but describe what they execute accurately.

## 2. Define the release you will build

Use one public product name consistently: RLCD. Credit ModernBERT and GLiClass prominently. Keep the repository name if desired. Describe this release as supervised fine-tuning with CE plus Brier and post-hoc calibration. The current training loop does not implement reinforcement learning, and a new foundation model has not been pretrained.

Your v0.1 promise is:

> Evaluate bounded choices locally, expose an explicit abstention outcome, and inspect the evidence behind confidence and performance claims.

Ship three connected artifacts:

1. A Python SDK that loads an exact versioned model bundle, validates schemas, and returns typed decisions with artifact and calibration metadata.
2. A static browser playground that performs actual local inference, lets users challenge it, and exports a decision receipt.
3. A benchmark runner that reproduces training comparisons, calibration results, failure cases, and deployment measurements.

Keep v0.1's supported candidate limit at 25 total outcomes, including abstention, unless you explicitly complete the capacity expansion experiment below. This makes at most 24 substantive choices available. Maintain `Score` and `Noul` interfaces, but mark their learned behavior experimental until each has separate evaluation evidence. Banking intent calibration does not validate a fraud score, cloud remediation, or any arbitrary proposition.

Your distinctive feature is the connection between a typed decision, its failure modes, its measured operating threshold, and its reproducible artifact. A familiar classifier with a different name is insufficient differentiation. Upstream [GLiClass already supports efficient single-pass dynamic-label classification](https://github.com/Knowledgator/GLiClass).

## 3. Execution rules and sequence

Complete the tasks in dependency order. You may parallelize independent documentation and fixture work, but give each worker exclusive file ownership. The coordinating agent owns integration, the evidence manifest, and the final review. Do not rewrite unrelated original research documents.

Preserve the current checkpoint, dataset files, and result reports as historical evidence. Write new runs into uniquely named directories. Record the source commit, dirty status, command, environment, inputs, outputs, exit status, and SHA-256 digests. Never overwrite a passing result with a different model under the same identity.

| Task | Depends on | Main owned files | Completion evidence |
| --- | --- | --- | --- |
| G0: Freeze the audit baseline | None | `reports/`, `WALKTHROUGH.md` | Reproduction report, claim corrections, hashes |
| G1: Repair SDK contracts | G0 | `core/`, `rlcd/`, `tests/` | Boundary and real-model integration tests |
| G1.5: Prove browser feasibility | G1 | Initial browser worker and export fixtures | Actual GLiClass graph execution and native parity on the target backend |
| G2: Build honest dataset splits | G0 | `scripts/download_real_benchmarks.py`, new `benchmarks/` | Source manifest, leakage audit, slice counts |
| G3: Train controlled baselines | G1, G1.5, G2 | `scripts/train.py`, benchmark adapters | Seeded runs and saved predictions |
| G4: Fit and evaluate calibration | G2, G3 | `core/calibration.py`, evaluation scripts | Frozen calibrators, risk/coverage report |
| G5: Export deployable bundles | G1, G4 | `export/`, bundle manifests, parity tests | Native/ONNX parity and measured precision tradeoffs |
| G6: Build the actual browser demo | G1, G5 | `webgpu-demo/` | Real browser inference, cross-language fixtures, receipts |
| G7: Add a fair generative comparison | G6 | browser comparator, benchmark adapters | Real outputs, schema validation, randomized serial timing |
| G8: Package and test a clean install | G1, G5, G6 | `pyproject.toml`, CI, README, release metadata | Fresh-environment smoke test and downloadable bundle |
| G9: Independent review and launch kit | G0 through G8 | reports, model card, launch drafts | Independent audit and claim-to-evidence map |

G7 can be omitted from the first release if the comparator is not ready. If you omit it, remove all speedup-versus-LLM claims. Real local inference and a useful abstention benchmark can stand on their own.

Before expanding G3 beyond pilot runs, complete G1.5 with the existing graph: load it in the target browser, run a real fixture, and compare its output to native inference. Reuse that worker in G6. This preliminary check establishes backend feasibility only; it does not establish release calibration or final performance. If the graph fails, resolve unsupported operations before spending time on the full training matrix. Preserve the GLiClass/ModernBERT identity. Record graph/runtime changes as deviations; any alternative model must be a separately named experimental bundle with its own evaluation, never a silent replacement behind the same UI or calibration claim.

## 4. G0: Freeze and correct the baseline

1. Record `fa70460a31c4716887c3b1d5bd1c1b73f367b8c1` and the current untracked changes. Preserve `Plans/implementation_plan.md` as user work. Treat document 08 as the engineering specification and this document as the correction plan.
2. Save the current weight, ONNX, tokenizer, calibration, and dataset digests. Preserve the old test as `legacy-random-5way-v1` in metadata without moving or deleting files that current scripts use.
3. Add `reports/claims.json`: each entry has a claim, status (`measured`, `unsupported`, `target`, or `historical`), artifact digest, dataset identity, command, and receipt path.
4. Correct `WALKTHROUGH.md`: remove claimed browser measurements, live integrations, guaranteed calibration, and zero compute cost. Label the current page a simulation until G6 passes. Correct parameter count and NLL. Preserve the narrow reproducible result and explicitly state candidate construction.
5. Do not call CE an improper loss. Both log loss and Brier are strictly proper under the usual predictive-distribution assumptions. Finite data, model misspecification, optimization, and distribution shift still affect calibration. A Brier term does not guarantee that a model refuses unknown questions.

Acceptance: the claims file contains no `measured` entry without a retrievable receipt. A reader can distinguish the existing Python result, the existing simulation, and future targets.

## 5. G1: Repair the SDK before retraining

### One model bundle and one formatter

Add a typed `ModelBundleManifest` with schema version, upstream model revision, weights digest, tokenizer/config digests, formatter version, backend/precision, max sequence length, maximum total candidates, supported primitives, calibration reference, and evaluated domain/schema scope. Use explicit local bundle loading. Keep upstream baseline loading as a separately named mode.

Create `core/formatting.py`. Move the exact question, context, label marker, and abstention rendering there. Training, evaluation, SDK, export fixtures, and browser golden vectors must derive from this contract. Distinguish canonical IDs from label descriptions. Do not silently rewrite descriptions or substitute a different abstention phrase.

Repair the injected-model path. Use an explicit tokenizer-to-tensors adapter rather than depending on an optionally initialized private pipeline member. Load the model in evaluation mode and use inference mode. Assert the output shape before mapping logits back to candidate IDs.

Before inference, validate candidate count and the complete encoded length. Reject overflow with a typed error that reports the limit and observed count. Do not silently truncate labels or evidence. A fixed-capacity graph may use padding and a valid-candidate mask; padding must never receive probability mass. For this bundle, reject 26 total candidates until a separately tested bundle supports them.

### Result and numerical contracts

Use bounded finite scalar types for probabilities and nonnegative latencies. Validate that each probability map has exactly the expected IDs, sums to one within a documented numerical tolerance, and agrees with the selected ID/probability. Reject malformed results instead of clamping or renormalizing arbitrary numbers. Softmax normalization of legitimate logits and zero probability on structural padding are valid operations.

Validate the calibration temperature as finite and strictly positive. Validate masks for shape, at least one valid outcome per row, and valid target membership. Reject NaN/Inf in valid logits. Represent ragged candidate sets with finite storage plus a mask; apply `-inf` only inside masked softmax/log-softmax. The current evaluator pads with `-inf` and then calls losses that reject nonfinite inputs, so mixed-cardinality evaluation needs a regression test.

Bind each calibration artifact to the weights/graph, formatter, tokenizer, precision, and tested scope. Report states such as `uncalibrated`, `validated_for_scope`, and `unvalidated_scope`. A weight mismatch must fail. A request outside an evaluated domain or schema must not inherit a broad calibration promise. Arbitrary custom browser text cannot be automatically certified as in-distribution.

Make `Noul.semantics` required. Preserve the v2 result definition: `p_true_given_sufficient_evidence = p_true / (p_true + p_false)` where the denominator is positive, with separate `p_insufficient_evidence`; use `None` when undefined or withheld under the declared abstention contract. `Score.expected_score` must clearly state conditioning on sufficient evidence. Neither value measures epistemic uncertainty by itself.

Preserve the semantics version in serialized Noul and batch results. The current training and evaluation paths only cover categorical Choice targets. Before promoting Score, add ordinal labels, ranked probability score, and MAE under the declared rubric. Before promoting Noul, add separate true/false/insufficient labels and conditional binary NLL/Brier on adjudicated sufficient-evidence cases, alongside the full three-outcome metrics. Do not fabricate these labels from the existing banking intent dataset.

Preserve custom table row keys, column names, types, row order, and reconstruction metadata. Encode wide rows as bounded header/value groups with row identity carried into each group. If lossless packing cannot fit the request, return a typed budget error or an explicitly counted continuation. Arithmetic, aggregation, and policy remain deterministic code. Test facts across local-attention boundaries and global layers; a 128-token local window does not mean the whole ModernBERT model sees only 128 tokens.

For the causal adapter, retain exact single-token and full-prefix continuation assertions. Decode with cleanup disabled, reject duplicate/special/multi-token slots, and test whitespace, punctuation, Unicode, and chat-template boundaries. Record full-vocabulary allowed-set mass and explicitly label slot probabilities as conditional on that set. Do not repair collisions through probability floors or caps. This adapter stays a comparison/compatibility path.

Keep DAG forward counts truthful: one batched call for independent queries, additional calls for dependent frontiers. Do not divide batch wall time by the number of fields and present that quotient as an observed per-query latency. Count every model invocation, including continuations.

Acceptance tests: 25 candidates succeed and 26 fail for the current bundle; mixed K values preserve masks and targets; injected model/tokenizer works; missing/mismatched bundles fail; nonfinite logits, invalid temperatures, impossible masks, and invalid result maps fail; omitted Noul semantics fails; custom row-key round trips are exact; blocked DAG descendants never dispatch.

## 6. G2: Build a benchmark that challenges the advertised behavior

### Data provenance and partitions

Pin raw dataset URLs to repository commits and save their byte hashes, source licenses, citations, source split names, and source row IDs. Banking77 contains 77 intents and 3,080 official test examples. CLINC supplies an explicit out-of-scope set. Cite the [Banking77 source](https://github.com/PolyAI-LDN/task-specific-datasets) and [CLINC source](https://github.com/clinc/oos-eval).

Do not merge CLINC train and validation blindly. Its standard OOS train and validation pools are small. Allocate disjoint source examples first, then generate candidate-set variants. For an initial protocol, use OOS train for training and partition OOS validation between development and calibration with exact counts recorded. Supplement calibration and development with disjoint banking missing-option examples. Do not claim a 20% OOS mixture when the source pool cannot supply it.

Split the Banking77 training pool into grouped, approximately stratified train/development/calibration partitions. Start with roughly 80/10/10, adjusted for duplicate groups and class coverage. Keep all transformations of one source utterance in the same partition. Exact normalized duplicates must not cross partitions. Audit near-duplicates and template families; fail the pipeline on known cross-partition contamination. Write files only after validation succeeds.

Preserve the official public test for comparison and disclose that 800 of its examples have already been inspected in this project. After reviewing this document, the legacy test is no longer a blind development target. Freeze all choices before running a new final evaluation. A separate, newly collected workflow set is needed before claiming general workflow transfer. Do not present public benchmark exposure as proof of novel real-world generalization.

The old synthetic `dataset_manifest.json` reports 211 contaminated test records. Keep these files as fixtures or historical development data. They cannot support release quality claims. Never exempt repeated abstention templates from leakage checks.

### Required evaluation slices

| Slice | Construction | What it measures |
| --- | --- | --- |
| Legacy random 5-way | Preserve the existing 1,000 examples | Reproduction only |
| Random candidates | K total in 3, 5, 9, 17, 25; true label present | Sensitivity to candidate count |
| Hard candidates | Use train-only confusion or frozen label-similarity neighbors | Distinguishing plausible alternatives |
| Missing correct option | Remove the true intent, add valid alternatives, audit overlapping meanings | Abstention when the available choices are inadequate |
| Distant OOS | Separate CLINC OOS source cases | Rejection of unrelated requests |
| Ambiguous or multiple intents | Independently labeled examples with an explicit target policy | Behavior when a single-choice schema does not fit |
| Order and wording perturbations | Fixed permutations and reviewed synonymous labels | Stability under presentation changes |
| Empty, malformed, and over-budget input | Contract fixtures | Typed rejection and policy safety |
| Tables | Preserve headers/types, place evidence across token boundaries | Preprocessing correctness and bounded semantic performance |
| Score and Noul | Separately labeled ordinal and true/false/insufficient examples | Primitive-specific quality, when claimed |

For K tests, define K as including abstention. Count unique source examples separately from generated variants. Use paired comparisons and resample by source utterance, not by variant, when computing intervals. You must not claim thousands of independent examples by reusing the same text with different distractors.

Use reviewed candidate definitions and legitimate mutually exclusive outcomes. A missing gold label does not prove that every remaining description is semantically false. Adjudicate ambiguous cases before sealing the benchmark. Keep the targeted 77-case diagnostic in the failure gallery regardless of later training improvements.

For workflow transfer, start with one narrow support-triage task. Create a labeling guide, obtain independent labels and disagreements, and keep the evaluation set separate from training. If only synthetic labels exist, call it a synthetic stress test. A Banking77 result alone does not validate refund authorization or security remediation.

Acceptance: publish actual train/dev/cal/test counts by source, class, primitive, K, and abstention subtype; verify no source identity crosses partitions; abort on insufficient pools; record unknown upstream pretraining overlap as a limitation.

## 7. G3: Run controlled training and useful baselines

Repair reproducibility first: seed Python, NumPy, and Torch; save seeds and environment; use a dedicated data-order generator; document nondeterministic accelerator operations. Reset gradient-accumulation groups correctly and normalize a final partial group by its actual size. Shuffle candidate order during training with a recorded seed. Save config and tokenizer with the checkpoint.

Do not tune model selection, candidate glossary, prompts, or loss weights on the final test. Select checkpoints on development metrics that include hard candidates and abstention. Fit temperature only after checkpoint selection on calibration data. Keep test evaluation read-only. Require an explicit output path and prevent silent overwrites or fallback to upstream weights.

Run these comparable configurations with the same data, descriptions, seeds, budget, and formatter:

1. Upstream GLiClass with T=1 and with separately fitted calibration.
2. Fine-tuned GLiClass with CE only, with and without calibration.
3. Fine-tuned GLiClass with CE plus Brier, with and without calibration.
4. A TF-IDF plus logistic-regression baseline for fixed Banking77 intent classification. Treat this as a task-specific baseline; it has a different dynamic-label interface. For open-set use, fit its rejection policy on development data and evaluate it separately.
5. The local generative model from G7, if a speed or quality comparison is published.

Start with one seed per learned configuration to debug the pipeline. For a public claim that Brier improves results, run at least three fixed seeds for CE and CE+Brier and show their variation. If the effect is unclear or CE wins, publish that result and choose the better supported model. Do not select only the most favorable seed or metric.

Use loss `mean(CE + lambda * sum_k((p_k - y_k)^2))`, with valid-candidate masking. Fix lambda before test evaluation. Positive weighted sums of these proper losses remain proper for the same target distribution, but they do not immunize the fitted network against overconfidence. Separate reweighted training distributions from the prevalence on which you calibrate and report results.

Save raw per-example logits, targets, masks, source IDs, candidate IDs/descriptions, and preprocessing hashes. These allow independent recalculation of every metric. Store training time, peak memory where measurable, selected checkpoint, and all trial identities.

The historical run reports 769.2 seconds for 2,200 examples and three epochs. Treat that as an old receipt, not an estimate for the larger matrix. Time a small pilot on the actual device, project the remaining cost, and report the assumptions before expanding runs.

## 8. G4: Calibrate, measure selective decisions, and choose policy

Fit positive scalar temperature through `T = exp(s)` by minimizing calibration NLL. Use stable logits and log-softmax. Remove the legacy probability floor in `scripts/train_calibrator.py`; calibrate raw logits instead of taking logs of already transformed probabilities. Consolidate the two calibration entry points so they cannot silently overwrite incompatible artifacts.

The standard temperature-scaling method is described in [Guo et al., 2017](https://proceedings.mlr.press/v70/guo17a.html). Evaluate it as a measured transformation. Never say that 92% confidence guarantees correctness on exactly 92% of arbitrary future requests.

Use scalar T as the default. If K-dependent miscalibration persists, compare a fitted positive function such as `T(K) = exp(a + b * log(K))` against scalar T using source-grouped, calibration-only cross-validation. Keep the simpler model unless the extra parameter improves held-out calibration folds with adequate coverage across K. Do not use an unfit `sqrt(log K)` rule, force top probabilities upward, or assume K>32 is a mathematical discontinuity. Temperature does not change the top-ranked answer and cannot create evidence missing from the candidate set.

Freeze the calibrator family after cross-validation, refit that family on the calibration partition, and keep the final test unopened throughout both steps. Final test outcomes must not select between scalar and K-dependent calibration.

Treat a future capacity expansion as a new bundle: inspect `max_num_classes`, distinguish tensor capacity from learned parameters, expand and validate the implementation, then test that expanded model at K=33 and K=78. Retrain and recalibrate across the sizes you support. Give the expanded bundle new graph, weight, and calibration digests and a separate release identity. The current 25-logit bundle is ineligible for these tests. Do not concatenate separately normalized chunk probabilities as if they form one globally calibrated distribution. Until this work passes, publish full-77 classification as unsupported.

### Required metrics and definitions

| Metric | Definition and reporting rule |
| --- | --- |
| Accuracy and macro-F1 | Report overall and in-scope intent-only values; disclose whether abstention is a class |
| Summed multiclass Brier | Mean over cases of the sum over valid outcomes; range 0 to 2; do not compare to a differently normalized Brier without explanation |
| NLL | Use log-softmax on the target; report invalid numerical cases as failures |
| ECE | Equal-width and equal-mass estimators with declared bin count, bin support, and uncertainty; do not select the flattering estimator |
| Abstention precision/recall | Separate distant OOS, missing-option, ambiguous, and in-scope false abstentions |
| Coverage | Fraction of requests accepted by the complete policy |
| Selective risk | Errors among accepted requests; include denominator and a one-sided 95% upper bound |
| OOS false acceptance | Fraction of expected-abstention requests that the policy accepts |
| Stability | Top-choice agreement and probability changes under permutations/label variants |
| Operational failures | Capacity/length errors, model-load failures, invalid outputs, parse failures, and timeouts |

Specify the acceptance rule before testing: the winner must be substantive, its probability must exceed the selected threshold, the schema/input must be valid, and the calibration scope must be eligible for the declared evaluation. Select the threshold on development data after fitting calibration. Freeze it before final evaluation. A threshold slider in the demo changes policy, not the model's probabilities.

Plot risk versus coverage, reliability bins, and performance by K. Include an always-abstain baseline so a low risk score cannot hide zero usefulness. Include false acceptance at the same coverage when comparing systems. For independent cases use a one-sided binomial bound such as Clopper-Pearson; for grouped variants use a source-level bootstrap and identify its limitations.

If the final test misses a gate, report the failure. A modified model may be evaluated as a new version, but the already inspected test is then a development/repeated-evaluation benchmark. Do not rerun changes against it and call every result a fresh holdout.

## 9. G5: Export and verify the deployable artifact

Make export require an explicit fine-tuned bundle. Verify the input digests before loading. Save the exact graph digest and graph inputs/outputs into the manifest. Fail on absent weights, wrong revisions, ONNX checker failures, or runtime validation failures. Do not swallow errors into a success report.

Start with the existing FP32 graph as a correctness reference. Test FP16 as the first browser optimization only if the complete graph and browser backend support it. Evaluate INT8 or another quantized representation as a separate experiment. The approximately 151M FP32 checkpoint is roughly 606 MB; 16-bit and 8-bit weight storage are roughly 303 MB and 151 MB before runtime/format overhead. Do not claim a 48 MB model without a different model or demonstrated compression.

Use a fixed-capacity 25-output graph if that is the most reliable export. Provide an explicit valid-candidate count/mask to postprocessing, assert capacity, and mask unused slots before normalization. Dynamic candidate dimensions are optional; silently unsupported dynamic candidates are unacceptable.

Build cross-runtime fixtures spanning batch sizes 1, 2, and 8; mixed K; K=3,5,9,17,25; short and long sequences; local/global attention boundaries; abstention; and near-ties. Test over-budget inputs as errors. Compare canonical token IDs, logits, probabilities, rankings, and policy decisions.

Add padding invariance tests: the same request with K=3,5,9,17 valid outcomes must agree between native evaluation and the fixed-capacity 25-output graph on valid logits, probabilities, ranking, and policy. Repeat alone and in a mixed-length/mixed-K batch. Structural padding must not introduce extra semantic labels or change the probability denominator.

For FP32 native versus CPU ONNX, start with maximum absolute logit error <=1e-4 on the defined fixture suite and probability error <=1e-5. These are engineering gates, not observed results. Investigate any failure before changing a tolerance. If a justified backend needs different tolerance, document it before final evaluation and include decision flips. Near-ties require explicit reporting rather than a blanket promise of identical rankings.

After selecting the final graph/precision, obtain calibration logits from that deployment path, fit its calibration artifact, and evaluate held-out quality. Reusing native calibration requires demonstrated numerical and decision equivalence plus a documented binding to the new graph. Quantization must not inherit a calibration label by filename.

Acceptance: publish the artifact sizes, hashes, supported dimensions, parity errors, decision disagreement counts, and quality deltas by backend. Release CI must fail if required artifacts are missing; local unit tests may explicitly skip heavy integration tests, but a skipped test is not release evidence.

## 10. G6: Build the browser playground around real inference

Use a small static app with a dedicated worker. Pin JavaScript dependency versions and commit the lockfile. Use Transformers.js for supported tokenizer/model operations and ONNX Runtime Web for the exact custom GLiClass graph when required. Do not substitute a generic zero-shot classification pipeline that computes a different model. The [Transformers.js WebGPU guide](https://huggingface.co/docs/transformers.js/en/guides/webgpu) documents GPU execution; graph support still needs verification on each target browser.

Suggested files: `webgpu-demo/src/worker.ts`, `bundle.ts`, `format.ts`, `inference.ts`, `metrics.ts`, and `main.ts`, plus shared JSON fixtures and browser tests. Avoid adding a server framework for local inference.

### Implement the actual path

1. Load the versioned manifest and show expected download size before loading weights. Fetch/cache the graph, tokenizer, and calibration artifact; verify their hashes with a practical streaming or incremental strategy so you do not create unnecessary copies of a 606 MB file.
2. Create a WebGPU session and execute a smoke fixture. Set the active-backend badge only after successful execution. If WebGPU fails, try WASM and label it accurately. If both fail, show a clear load error and a retry path; do not display fixture predictions.
3. Reproduce canonical formatting and tokenizer outputs. Validate every request, candidate count, and token budget before inference. Keep UI inputs disabled only during operations that require it; provide cancel/reset for load and run state.
4. Run the graph in the worker. Await output readback before stopping inference timing. Apply the bound temperature and stable masked softmax. Send typed results and timing breakdowns to the UI.
5. Render all candidate probabilities, the separate abstention result, and the deterministic policy action. Indicate when a custom schema or text falls outside established calibration evidence. Do not label every custom confidence bar calibrated.
6. Export a JSON receipt containing bundle/hash identity, backend, browser/device information available with consent, token and candidate counts, probabilities, calibration status, timings, and errors. Include the user's input text only when they explicitly choose it. Keep inference inputs local by default.

### Make the demo worth sharing

Use three visible interactions:

1. Paste a support request and edit its candidate actions. The result must respond to the actual input.
2. Remove the expected answer or add a confusing alternative. Show how probabilities and abstention change, including failures.
3. Move the policy threshold and inspect accepted/error/abstention tradeoffs from a named precomputed benchmark. Label aggregate benchmark charts separately from the current live inference.

Add a small failure gallery, with the artifact version and expected outcome for each case. Include the existing missing-option failure. A failure gallery makes the boundaries inspectable and gives contributors concrete work.

The browser should demonstrate routing to a review queue or a local dry-run action. Use a local action log for the demo. Do not describe a string as a refund, webhook delivery, or Slack notification. Model classification of a support intent is insufficient authority to move money.

### Measure browser performance

Publish cold download, initialization/compilation, tokenization, completed inference/readback, postprocessing, and end-to-end request time separately. Measure warm performance after at least 20 warmups and 200 actual executions per specified shape. Report p50/p95/p99 and failures, not a single fastest sample. Include token length, K, query count, batch size, graph digest, dtype, browser version, OS, hardware, and active backend.

Measure on Chrome and Safari separately, plus an actual WASM fallback run if claimed. Platform support for WebGPU does not prove support for this graph. If only Chrome passes, publish Chrome support and report Safari as unverified or unsupported. Do not present MPS training latency as WebGPU latency.

Use an aspiration of warm p50 <=30 ms and p95 <=60 ms for a declared short-input shape on the chosen reference laptop. These are engineering targets. Publish the measured outcome if slower; do not manufacture sub-10 ms or 15 ms numbers. A useful release can be slower than the aspiration.

Treat the first download as a product constraint. Record bytes transferred and time to first successful inference on a declared connection, including an empty cache. Show progress and permit cancellation. Target a browser bundle below 200 MB only if a measured quantized or smaller-model variant retains the frozen quality gates. If the validated bundle remains 303 or 606 MB, disclose that size before download and avoid calling startup instant. Do not compress the headline at the expense of the evaluated behavior.

Acceptance: editing context changes the token IDs sent to the worker; loading a different bundle changes the recorded digest; losing WebGPU produces a real fallback/error; every displayed prediction traces to an executed graph; receipts reproduce the UI values; no inference timer or output depends on a preset probability array.

## 11. G7: Compare against real JSON generation

Use a small, licensed browser-compatible causal model with a pinned revision. A candidate is `onnx-community/Qwen2.5-0.5B-Instruct`; verify its actual files, license, compatible runtime, and memory requirements before adopting it. Treat this as a candidate to validate, not an already working integration.

Give both systems the same input text and candidate semantics, with explicit abstention. For the generative model, use a compact instruction to emit the selected ID in a minimal JSON object. Use deterministic decoding, a declared maximum token budget, and the exact prompt in the report. Do not inflate the baseline with verbose explanations, artificial network delays, or pretty-printed JSON.

Measure first-token time, full valid-result time, actual token IDs/count, parse/schema validity, semantic accuracy, abstention, and timeout rate. Count malformed outputs as failures. Do not treat a generated confidence number as empirically calibrated. Apply a predeclared timeout and report timeout failures separately rather than excluding them from latency/quality tables.

Run models serially with randomized order during timed comparisons to avoid GPU contention. Side-by-side visual display may show the two completed outputs, but its animation is not the benchmark. Disclose separate downloads and memory use. Allow the heavier comparator to be opt-in.

Optionally compare one-pass causal slot selection on the same causal model to its own JSON generation. This isolates generation overhead more clearly, although it still changes the decision interface. Do not attribute every speed difference between differently sized models to autoregression alone.

Acceptance: a benchmark receipt contains real input/output tokens, validated results, failure counts, and full timing distributions for both models. Publish speed ratios only for explicitly named task, device, model, precision, and quality conditions.

## 12. G8 and G9: Package, verify independently, and decide release readiness

Declare all direct runtime dependencies and use backend extras where useful. The current package imports `gliclass` without declaring it. Verify supported Python and dependency versions instead of retaining broad version floors that predate ModernBERT. Pin reproducible development/training environments and record JS/runtime versions.

Add a code license chosen by the maintainer, third-party notices, data citations, a model card, a dataset card, a changelog, and contribution instructions. Preserve upstream model license obligations. Publish large weights through a versioned model/artifact host with checksums and a download command; keeping them out of Git is appropriate, but a fresh clone still needs a reproducible acquisition path. External publication remains a separate release action for Dante.

Create an actual ten-line quickstart from the final API. Run it in a fresh environment outside the repository checkout. Test the first download and an offline cached run. Restrict normal pytest discovery to first-party tests so bundled reference repositories do not break the default command. CI should include typing, schema tests, a tiny CPU integration path, and a separately identified artifact/browser release job.

Before final evaluation, freeze a release candidate commit, bundle digests, calibration protocol, policy threshold, and benchmark configuration. Run measurements against that exact commit and those artifacts. If source changes, state whether the affected checks were repeated. Evidence reports may be published afterward while retaining the measured source commit identity.

Have an independent reviewer inspect the diff, manifests, raw predictions, and receipts. Ask them to attempt missing artifacts, wrong calibrators, excessive K, long inputs, missing gold options, changed label order, invalid outputs, and browser fallback. The reviewer must not be the implementation author. If independent review is unavailable, mark it unavailable rather than representing self-checks as independent approval.

### Release gates

| Gate | Pass condition | If it fails |
| --- | --- | --- |
| Evidence integrity | No simulated outputs/timings presented as measurements; every headline maps to a receipt | Correct claims before any launch |
| Correctness | No silent model fallback, candidate loss, invalid probability output, or calibration identity mismatch | Fix before SDK release |
| Browser truthfulness | Actual inference on each advertised backend, parity fixtures, completed readback timing | Remove backend/speed claims or defer browser launch |
| Reproducibility | Clean install and model acquisition work; pinned manifests and public prediction receipts reproduce metrics | Repair packaging/evidence |
| Calibration claim | Publish Brier, NLL, both ECE variants, support, scope, and uncertainty; target equal-width ECE <0.04 on the declared distribution | Publish as experimental if calibration targets are missed |
| Useful selective policy | On a preregistered task, at least 200 accepted independent cases, coverage >=0.50, and one-sided 95% selective-risk upper bound <=0.05 | No reliable-automation claim for that task |
| Distant OOS handling | At least 200 independent distant-OOS cases; target false-acceptance upper bound <=0.05 | Show the failure; do not generalize beyond this slice |
| Missing-answer handling | At least 200 independently sourced, adjudicated missing-option cases; target false-acceptance upper bound <=0.05, reported separately | No reliable missing-answer rejection claim; retain the stress demo and label its limitation |
| Ambiguity handling | Separate false-acceptance results and denominators; label exploratory unless it has its own preregistered policy and sufficient independent test support | Do not pool it into a distant-OOS success claim |
| Classification quality | For the declared slice, target accuracy >=0.80 and macro-F1 >=0.75; compare useful baselines | Revisit data/model or release a benchmark contribution |
| Independent review | No unresolved P0/P1 findings and review receipts identify the reviewed commit/artifacts | Hold the affected release claims |

These thresholds are proposed project gates, not universal standards or achieved results. Freeze them before final evaluation. Apply them to each domain/primitive you advertise. Sparse slices remain unvalidated even if the pooled result passes.

### Proposed commands and output contract

Implement and document commands with these capabilities; these examples are a target CLI, not commands that work today:

```text
python -m benchmarks.prepare --config configs/benchmark-v1.yaml --out runs/<run>/data
python -m benchmarks.audit_data --manifest runs/<run>/data/manifest.json
python -m scripts.train --config configs/ce-brier-v1.yaml --run-dir runs/<run>
python -m benchmarks.predict --bundle <bundle> --split <split> --out <predictions>
python -m benchmarks.fit_calibration --predictions <calibration-predictions> --out <calibrator>
python -m benchmarks.evaluate --predictions <test-predictions> --policy <frozen-policy> --out <report>
python -m export.export_onnx --bundle <bundle> --out <deployment-bundle>
python -m benchmarks.verify_bundle --bundle <deployment-bundle>
python -m benchmarks.verify_claims --claims reports/claims.json
```

Each run directory must contain `manifest.json`, `environment.json`, source/split hashes, training configuration, model-selection history, calibrator identity, raw predictions, `metrics.json`, `latency.json`, plots, failure examples, and a short report. Use stable IDs so another person can trace a point on a chart to its source case.

Gemini's final handoff must state what changed, the exact commit and artifacts tested, commands with exit statuses, failed or skipped checks, metrics versus gates, unresolved limitations, and a deviations table. A list of files created and `25 tests passed` is insufficient completion evidence.

## 13. What to show on GitHub

Rewrite the README around the working user flow. Put a short recording of actual inference near the top, followed by a runnable quickstart, a small measured benchmark table, installation/model-download instructions, and clear limitations. Move the long research-document index lower on the page. GitHub's [README guidance](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-readmes) emphasizes what a project does, why it is useful, and how to get started.

Make the release easy to evaluate:

- A browser link works without an API key and clearly discloses the model download.
- A developer can reproduce the headline metric with one documented command after installation.
- Each model/runtime has a downloadable manifest and prediction receipt.
- The failure gallery includes cases the current model gets wrong.
- One real support-triage example connects a typed decision to a deterministic local policy.
- Upstream attribution and the limits of the contribution are prominent.
- Contribution issues are specific: reproduce on a device, add an adjudicated stress case, improve packaging, or fix a demonstrated failure.

Do not lead with a Jev replacement claim, a novel-foundation-model claim, autonomous money movement, universal calibration, or an unmeasured latency multiplier. The repository becomes more useful when developers can compare their own model and decision policy with the same benchmark contract.

### A practical adoption loop

Before launch, ask a small number of willing external developers to install the release candidate and try their own cases. Count completion, time to first real inference, confusing steps, and reproducible bugs. This plan does not authorize automated outreach or messages; Dante chooses who to contact.

After launch, prioritize installation failures and contributions that improve reproducibility. Publish fixes and exact quality/performance changes. An independent benchmark reproduction or a real integration is stronger adoption evidence than a burst of stars.

Track daily aggregate GitHub views/clones, stars, forks with substantive work, model downloads, independent reproductions, and useful issues for the first 14 days where those signals are available. Treat stars as an attention measure. Do not claim a causal growth formula or collect browser text for analytics. Optional demo telemetry must be disclosed and exclude input content by default.

## 14. What to post

Prepare the copy after the final report exists. Replace every placeholder with a number from a named, frozen receipt. If a gate fails, use the experimental-results version and state the failure. Do not publish these drafts automatically.

### Launch post after the real browser and evidence gates pass

> I built RLCD: a small decision model you can run in your browser.
>
> Change the choices. Remove the right answer. See when it abstains.
>
> [p50] ms median on [device]. Open weights, benchmarks, and failure cases.
>
> Try breaking it.

Attach a 20 to 30 second screen recording of the actual release build. Use the first reply for the repository, demo, exact benchmark report, upstream credit, and the task scope. Check the final post length after replacing placeholders.

### First reply

> Repo: [link]
> Demo: [link]
> Results and reproduction: [link]
>
> Built on ModernBERT + GLiClass. The current evaluation covers [scope], with [N] source examples. Calibration and abstention have measured limits; the failure gallery is public.

### Technical follow-up after controlled evaluation

> Our first result was 96% accurate on five-option banking questions.
>
> Removing the labeled answer exposed the weakness: the model abstained on only 16 of 77 diagnostic cases.
>
> The release now publishes missing-option tests, risk versus coverage, and the remaining failures: [link]

Only use the last sentence once those artifacts are published. If you report improvement from the diagnostic, disclose that those 77 cases informed development and show a separately frozen evaluation for the improvement claim.

### If the model misses the selective-risk target

> I built a browser playground for a question that accuracy misses: when should a decision model refuse to choose?
>
> The benchmark tests missing answers, confusing alternatives, and confidence under changed choices.
>
> Results, code, and failures: [link]

Use this version only once the browser and benchmark actually work. It is a useful research release even if the model has unresolved quality limitations.

### Recording storyboard

| Time | Screen action | Evidence shown |
| --- | --- | --- |
| 0 to 5 seconds | Enter a new support request | Actual editable input and model/backend identity |
| 5 to 11 seconds | Run inference | Real candidate probabilities, abstention, measured end-to-end time |
| 11 to 18 seconds | Remove or change a candidate and rerun | Changed output, including an honest failure if it occurs |
| 18 to 24 seconds | Move the policy threshold | A deterministic accept/review decision with unchanged probabilities |
| 24 to 30 seconds | Open/download the receipt | Hashes, scope, test report link, repository URL |

Keep load-time disclosures visible in the page. Editing a recording for brevity is acceptable when cuts are obvious; do not splice outputs to imply continuous performance or hide failure as success. For a static image, export the actual risk/coverage plot and a small measured performance table with device, K, sequence length, and sample counts.

## 15. Where to stop

Finish the truthful SDK, real browser inference, hard abstention evaluation, reproducible release bundle, and concise launch material before attempting a larger foundation model, multi-domain training program, live financial execution, or speculative infrastructure.

If users can install it, challenge it, inspect its failures, and reproduce its measurements, you will have a defensible open-source contribution. Large adoption remains uncertain. The execution plan should improve the product and the evidence even if the launch attracts modest attention.
