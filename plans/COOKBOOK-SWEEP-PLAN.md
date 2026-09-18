# Cookbook sweep plan

Goal: run the shipped OpenJev checkpoint against as many cookbook use cases as
can be reproduced locally, log every result, then promote the winners into the
README and the `index.html` demo.

## The measurement that sets the budget

Batched throughput on this machine, measured:

```
device: mps       load + move: 2.6s
batch=16 -> 220.7 items/sec
batch=32 -> 235.3 items/sec
```

Twelve tasks at 1,000 items, two checkpoints, three label-framing arms is 72,000
forward passes, which is **about five minutes of GPU time**. Inference is not the
bottleneck. The bottlenecks are dataset downloads, repeated model loading, and
re-running the model every time an analysis changes.

The architecture below removes all three. Breadth is effectively free, so test
everything plausible rather than picking four.

---

## Architecture

Three stages. Do not write one script per experiment; that reloads 605 MB of
weights twelve times and makes every analysis change a full re-run.

```
scripts/gen_suite/
  tasks.py          task registry, one loader per task -> list[TaskItem]
  prepare_all.py    stage 1: parallel dataset fetch  -> data/gen/<task>.jsonl
  run_inference.py  stage 2: load ONE checkpoint, run ALL tasks batched -> .npz
  analyze.py        stage 3: pure numpy over cached logits -> reports/v2/*.json
  render.py         stage 4: tables into README via existing markers
```

`TaskItem` is the single shared shape:

```python
{"task": str, "id": str, "question": str, "context": str,
 "candidates": [{"id": str, "description": str}],
 "target_id": str, "meta": {...}}
```

### Stage 1, parallel

`prepare_all.py` fetches every dataset concurrently with a `ThreadPoolExecutor`,
since this is network bound. It writes one JSONL per task and one
`data/gen/manifest.json` recording, per task, the source, the sampled row ids,
the seed, and the item count. Nothing downstream re-downloads.

### Stage 2, sequential per checkpoint, batched

`run_inference.py --checkpoint {base|finetuned}` loads the model once and runs
every task in one pass, batch size 32, sorted by token length to minimise
padding. It writes raw logits to
`reports/v2/_cache/logits_<checkpoint>_<task>.npz` holding `item_ids`,
`logits` (N x K), and `target_idx`.

Run the two checkpoints one after the other. They contend for the same GPU, so
parallelising them makes both slower.

**Cache the logits.** Every metric downstream is a function of logits, so once
this stage completes, accuracy, ECE, Brier, cascade curves, threshold sweeps, and
any new metric invented later are instant and need no GPU.

### Stage 3, parallel, instant

`analyze.py` reads the caches and computes everything with numpy, parallel across
tasks with a `ProcessPoolExecutor`. Reuses `core/calibration.py::compute_ece`.
Writes one receipt per task plus `reports/v2/gen_sweep_summary.json` holding the
full cross-task, cross-checkpoint matrix.

### Smoke first

Every script takes `--smoke`, which caps each task at 20 items. Run the entire
pipeline end to end in smoke mode before the full run. It finishes in under two
minutes and catches every shape error, missing column, and prompt bug. Do not
start a full run until smoke is green.

---

## Task registry

All data sources verified reachable. Every task is a `Choice` question with
K <= 25, which is the only primitive the repository has benchmarked.

| ID | Task | Cookbook | Source | K | Demo value |
| :--- | :--- | :--- | :--- | ---: | :--- |
| T01 | Phishing vs legitimate email | `patterns_confidence_routing` | `ealvaradob/phishing-dataset`, `SetFit/enron_spam` | 3 | **High**, matches the live trend |
| T02 | Jailbreak vs benign prompt | `llm_guardrails` | `TrustAIRLab/in-the-wild-jailbreak-prompts`, `tatsu-lab/alpaca` | 3 | **High**, instantly legible |
| T03 | Hazard severity routing | `llm_guardrails` | same as T02, 4 routing outcomes | 5 | Medium |
| T04 | Shopify product taxonomy, flat leaf | `hierarchical_classification` | Shopify `categories.txt` | 25 | Medium |
| T05 | Shopify taxonomy, hierarchical beam | `hierarchical_classification` | same | <=25/node | **High**, breaks the 25 ceiling |
| T06 | Banking77 in-domain control | n/a | existing `data/real_banking_test.jsonl` | 5 | Baseline anchor |
| T07 | CLINC150 intent routing, 150 intents | `patterns_intent_routing` | CLINC150, already local | <=25/node | High |
| T08 | Smart home command interpretation | `demos_smart_home` | authored fixture | 8 | **Highest demo value** |
| T09 | Function and tool routing | `function_calling` | authored fixture, 20 typed functions | 21 | **High** for developers |
| T10 | Agent skill selection | `skill_suggestion` | Nous Hermes roster, GitHub | <=25/node | Medium |
| T11 | Passage relevance re-ranking | `rerank_typesafe` | BEIR or CLERC subset | 10 | Medium |
| T12 | Prompt-injection screening in RAG | `classifying_rag_passages` | injection corpus plus clean docs | 3 | High |

Authored fixtures (T08, T09) ship with a provenance manifest marking them
project-authored, following the convention in `oss/open-jev-v2/openjev`.

Excluded: `citation_check`, `date_extraction`, `entity_alignment`, `autoformat`,
`semantic_find`, `autoresearch_feature_discovery`, `consistency_*`. Each needs
either bespoke data or the `Score` and `Noul` primitives, which the repository
documents as experimental and unbenchmarked.

---

## Arms

Every task runs across the full cross product:

- **Checkpoints**: `base` (`knowledgator/gliclass-modern-base-v2.0`) and
  `finetuned` (`artifacts/v2/model.safetensors`).
- **Label framing**: `neutral`, `verbose`, `banking_framed`.

The framing arm matters because of a 6-item probe on authored phishing email
where base scored 5/6 and the fine-tune scored 1/6, abstaining on five at
confidence 0.58 to 0.80. If the fine-tune recovers under banking framing, the
cause is a learned domain gate rather than lost capability, which is a real and
publishable mechanism. All three arms get reported regardless of outcome.

---

## Metrics, per task per arm

Computed once from cached logits:

- accuracy, NLL, Brier, ECE equal-width and adaptive, MCE with `min_bin_count=10`
- abstention rate, precision, recall
- **over-abstention rate**: share of items where the model abstained although the
  correct option was present
- selective-risk curve, thresholds 0.50 to 0.99 step 0.01, with coverage,
  retained accuracy, and errors above the gate
- 1000-sample bootstrap 95% CI on accuracy and abstention rate
- measured p50 and p90 latency per item at that task's K

---

## Promotion rules

After the sweep, `analyze.py` emits a ranked table. A task is demo-ready when it
clears all of:

1. Accuracy at or above 85% on at least one checkpoint.
2. ECE adaptive at or below 5% on that checkpoint.
3. Over-abstention rate at or below 10%.
4. At least 500 evaluated items.
5. A viewer can understand the task from one sentence.

Promote the top result into `webgpu-demo/index.html` as a new preset group, with
its candidate descriptions taken verbatim from the task registry so the demo runs
the benchmarked configuration.

Promote the top three into the README through `render_receipts.py`, and commit
the full twelve-task matrix as a linked table so every number that exists is
reachable. Leading with the best result is fine. Deleting the others is not, and
it is also unnecessary, since the full matrix is what makes the best number
credible.

If the winning checkpoint is `base` rather than `finetuned`, ship both and say
which is which. A general checkpoint that works across nine tasks is a better
product than a banking specialist that works on one.

---

## Runtime estimate

| Stage | Time |
| :--- | :--- |
| Stage 1, parallel downloads | 3 to 8 min, network bound |
| Smoke run, 20 items per task | under 2 min |
| Stage 2, both checkpoints, all arms | 5 to 12 min |
| Stage 3, analysis | under 1 min |
| Stage 4, render plus demo wiring | 20 min |

Whole sweep under an hour, dominated by downloads and wiring rather than compute.
