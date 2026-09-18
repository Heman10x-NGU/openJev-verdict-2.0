You are working in a public open-source repository at:

/Users/heman10x/Downloads/claude_dev/personal_projects/Mind-Palace/RLCD-demo

Read `Plans/COOKBOOK-SWEEP-PLAN.md` first. It is the specification. Build the
three-stage harness it describes, run the twelve-task sweep, and promote the
winners into the README and the browser demo.

## What this repository is

OpenJev, a 151M ModernBERT plus GLiClass encoder that answers typed decision
questions in one forward pass. Given text plus a list of candidate option
descriptions, it returns one calibrated probability per candidate plus an
explicit abstention option. The head produces 25 logits, so each question takes at
most 24 substantive candidates plus abstention.

Two checkpoints:
- `base`: `knowledgator/gliclass-modern-base-v2.0`, already in the HuggingFace
  cache, a general zero-shot classifier.
- `finetuned`: `artifacts/v2/model.safetensors`, post-trained on PolyAI Banking77.

Existing machinery to reuse rather than rewrite:
- `core/formatting.py::format_prompt` builds every prompt. Do not write new prompt
  construction anywhere.
- `core/calibration.py::compute_ece` does all binning.
- `scripts/render_receipts.py` renders JSON receipts into README tables between
  `<!-- BEGIN GENERATED: name -->` markers.

## Goal

Find which cookbook use cases this model does well, with evidence. Log every
result. Then put the best one in the demo and the best three in the README.

## Efficiency requirements, these are the point

Measured on this machine: batch 32 on MPS runs 235 items/sec, and a checkpoint
loads in 2.6 seconds. Inference is cheap. The waste is elsewhere, so:

1. **Do not write one script per experiment.** Twelve scripts each loading 605 MB
   is the slow design. Build the shared three-stage harness in
   `scripts/gen_suite/`.
2. **Load each checkpoint exactly once** and run all twelve tasks and all three
   framing arms against it in a single process.
3. **Batch at 32, sorted by token length** to cut padding waste.
4. **Cache raw logits to .npz.** Every metric is a function of logits. Once stage
   2 finishes, all analysis is numpy and needs no GPU. If you change a metric, do
   not re-run the model.
5. **Parallelise stage 1** dataset downloads with a ThreadPoolExecutor, since
   they are network bound. Parallelise stage 3 analysis with a
   ProcessPoolExecutor.
6. **Do not parallelise the two checkpoints.** They contend for the same GPU and
   both get slower.
7. **Run `--smoke` first**, 20 items per task, end to end through all four
   stages. It finishes in under two minutes. Do not start the full run until
   smoke is green. This is the single biggest time saver available.

## Hard rules

1. Never run `git merge`, `gh pr merge`, or a rebase that lands one branch on
   another. Never push to `main` or `master`. Create a feature branch and push
   only that. Pushing the branch is the stop point.
2. Never hand-write a metric into Markdown, HTML, or JavaScript. Every number a
   human reads comes from a JSON receipt rendered by `render_receipts.py`. If you
   are typing a percentage, you are doing the wrong thing.
3. Never invent a number. If a stage has not run, its receipt does not exist and
   the number goes nowhere. Report what commands actually printed, including
   crashes.
4. Show real output. Do not summarise a run as passing. Paste the tail.
5. **Never tune anything after seeing a result.** Do not adjust a prompt, a
   threshold, a label description, or a sample because a number disappointed you.
   The three label-framing arms are declared up front and all three get reported.
   Changing a setting to improve a number after the fact is fabrication and
   invalidates the sweep.
6. Every task reports both checkpoints. A task reporting one checkpoint is
   incomplete and must be rerun.
7. Commit the sampled row ids and the seed for every task, in
   `data/gen/manifest.json`.
8. Keep changes surgical. Do not refactor adjacent code. Do not regress the
   existing 42 tests or `scripts/render_receipts.py --check`. Do not modify,
   rename, or delete any existing receipt under `reports/v2/`.
9. Leave `oss/`, `files-not-tobe-commited/`, and `RLCD Cookbook/` untouched and
   uncommitted.
10. Prose style: active voice, second person, present tense, sentence-case
    headings, Oxford comma. No em dashes. No "not X but Y". No bold-label
    bullets. No Summary section unless asked.

## Data sources, all verified reachable

- `ealvaradob/phishing-dataset`, `SetFit/enron_spam`
- `TrustAIRLab/in-the-wild-jailbreak-prompts`, `tatsu-lab/alpaca`
- `https://raw.githubusercontent.com/Shopify/product-taxonomy/main/dist/en/categories.txt`
- Nous Hermes skill roster from GitHub
- CLINC150 and Banking77, already local under `data/`
- T08 and T09 are fixtures you author, with a provenance manifest marking them
  project-authored

## Order, stop and report at each marker

1. Build `scripts/gen_suite/` stages 1 through 4. Run `--smoke` end to end.
   **STOP.** Paste the smoke output and the shape of one cached .npz.
2. Run stage 1 full. **STOP.** Report item counts per task and the manifest.
3. Run stage 2 for both checkpoints, all arms. **STOP.** Report wall-clock per
   checkpoint and total cached logit rows.
4. Run stage 3. **STOP.** Paste the full twelve-task, two-checkpoint,
   three-arm matrix, plus the ranked promotion table showing which tasks cleared
   all five promotion gates.
5. Wire the top task into `webgpu-demo/index.html` as a new preset group, using
   candidate descriptions verbatim from the task registry. Render the top three
   into the README. **STOP.** Report what was promoted and why.

## Expect this

The fine-tuned checkpoint may lose to the base checkpoint on most out-of-domain
tasks. A 6-item probe on authored phishing email scored base 5/6 and fine-tuned
1/6, with the fine-tune abstaining on five at confidence 0.58 to 0.80. If the
sweep confirms this at scale, report it plainly and promote the base checkpoint
where it wins. That is a result, not a problem to engineer around.

## Do not

Do not write launch copy, tweets, or thread drafts.

Do not showcase the `Score` or `Noul` primitives. The repository documents them as
experimental and unbenchmarked. Every task in the registry is a `Choice`.

Do not delete losing results. Promote the winners, and commit the full matrix as a
linked table so every number that exists stays reachable.

## Done means

- `scripts/gen_suite/` exists with all four stages and a working `--smoke` flag.
- `data/gen/manifest.json` records source, row ids, seed, and count per task.
- `reports/v2/gen_sweep_summary.json` holds the full matrix.
- One receipt per task exists under `reports/v2/`.
- The README renders the top three and links the full matrix.
- `webgpu-demo/index.html` carries the winning task as a preset group.
- `CI_REQUIRE_ARTIFACTS=1 pytest tests/ -q` passes with no skips.
- `python scripts/render_receipts.py --check` exits zero.
- The feature branch is pushed and `main` is untouched.
