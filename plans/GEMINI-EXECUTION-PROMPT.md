You are working in an existing Python and JavaScript research repository at:

/Users/heman10x/Downloads/claude_dev/personal_projects/Mind-Palace/RLCD-demo

Read `Plans/LAUNCH-FIX-PLAN.md` completely before doing anything. Parts A
through D explain why the work is scoped the way it is. The numbered phases are
the specification. Execute them in order.

## Fill these in first

PROJECT_NAME = <new name, must not contain "jev">
HF_REPO_ID = <HuggingFace repo id for the weights>

If either is blank, stop and ask. Do not invent one.

## What this repository is

A non-autoregressive decision engine on ModernBERT plus GLiClass-v2, 151M
parameters. It takes text plus a list of candidate option descriptions and
returns one probability per candidate in a single forward pass, including an
explicit abstention option. It was evaluated on PolyAI Banking77 and CLINC150
out-of-scope data. A browser demo runs the exported ONNX model client-side
through ONNX Runtime Web.

## What an adversarial review found

The published numbers, the code, and the browser demo disagree with each other.
The browser builds a different prompt than the model was trained on. The weights
are gitignored so nobody who clones the repository can run anything. Several
headline claims have no reproducible receipt. The benchmark has a confound: only
53 of 77 category labels have a curated glossary description and the rest get a
template fallback, so part of the accuracy is measuring glossary coverage. And
there is no control baseline of any kind, so nothing currently establishes that
the model reads the input rather than exploiting label priors.

Three competing open-source projects sit in `oss/` for reference. Read their
approaches if useful, but never modify anything under `oss/` or under
`files-not-tobe-commited/`.

## Hard rules

1. Never run `git merge`, `gh pr merge`, or a rebase that lands one branch on
   another. Never push to `main` or `master`. Work on a feature branch. Pushing
   the branch is the stop point.
2. Never hand-write a metric into a Markdown, HTML, or JavaScript file. Every
   number a human reads is generated from a JSON receipt under `reports/v2/`. If
   you are typing a percentage, you are doing the wrong thing.
3. Never invent a number. If a script has not run, its receipt does not exist and
   the number goes nowhere. Report what commands actually printed, including
   failures.
4. Show real output. Do not summarise a test run as passing. Paste the tail.
5. Publish results that are worse than the current claims. Several experiments in
   Phase 3 are designed to find weaknesses. Finding one is success, not a problem
   to work around. Never tune an experiment until it produces a better number.
6. Keep changes surgical. Do not refactor adjacent code, reformat untouched
   files, or improve anything the plan does not name.
7. Prose style: active voice, second person, present tense, sentence-case
   headings, Oxford comma. No em dashes. No "not X but Y" constructions. No
   bold-label bullets. Explain jargon in parentheses on first use. No Summary or
   Conclusion section unless asked.

## Order

Phase 0, decisions. Phase 1, the four blockers: weight hosting, the shared prompt
contract between Python and the browser, the worker memory leak and races and
false backend label, and removing hardcoded metrics from the demo. Phase 2,
correcting the record: checkpoint selection on validation NLL, calibration
reported in both directions, abstention precision beside recall, failure gallery
regenerated. Phase 2 requires a retrain, roughly 891 seconds on Apple Silicon
MPS. Phase 3, the nine experiments. Phase 4, rename. Phase 5, generate all
documentation from the receipts. Phase 6, verification from a fresh worktree.
Phase 7, stop. Do not write launch copy.

## Stop and report at these points

- After 1.2, show the browser parity test output.
- After 2.1, show the per-epoch metric table and which epoch the NLL criterion
  selected.
- After each Phase 3 experiment, show the number. State explicitly whether it is
  better or worse than the current published claim. Expect E1 through E5 to be
  worse.
- After Phase 6, paste the verification output verbatim.

## Done means

- Fresh clone plus `python scripts/download_artifacts.py` plus
  `CI_REQUIRE_ARTIFACTS=1 pytest tests/ -q` passes with no skips.
- `python scripts/render_receipts.py --check` exits zero.
- `grep -rn "1\.59\|93\.70\|21\.46" README.md WALKTHROUGH.md webgpu-demo/index.html`
  returns nothing.
- Every abstention recall figure has a precision figure beside it.
- Every latency figure has a percentile label and a source receipt.
- The browser and `scripts/evaluate.py` agree to within 1e-3 on the same input.
- Switching demo checkpoints ten times does not grow memory beyond one session.
- `reports/v2/` contains a receipt for every one of the nine Phase 3 experiments.
