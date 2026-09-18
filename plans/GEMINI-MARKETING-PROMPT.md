You are working in a public open-source repository at:

/Users/heman10x/Downloads/claude_dev/personal_projects/Mind-Palace/RLCD-demo

Remote: https://github.com/Heman10x-NGU/Verdict-open-jev
Weights: https://huggingface.co/heman10x/rlcd-modernbert-151m

Read `Plans/LAUNCH-MARKETING-PLAN.md` first. It is the specification. Execute
Track A completely, then Track B. Stop before Track C.

## What this repository is

A 151M ModernBERT plus GLiClass encoder fine-tuned to answer typed decision
questions in one forward pass. Given a text input and a list of candidate option
descriptions, it returns one calibrated probability per candidate plus an
explicit abstention option. Evaluated on PolyAI Banking77 and CLINC150. A browser
demo runs the exported ONNX model client-side through ONNX Runtime Web.

The repository is already public and its scientific receipts are in good shape:
42 tests pass, `scripts/render_receipts.py --check` confirms every published
table matches its committed JSON, and nine experiment receipts live under
`reports/v2/`. Do not regress any of that.

## Why this work exists

The audience discussing this class of model is adopting one pattern: run a fast
calibrated classifier, gate on its confidence, and escalate the uncertain cases to
a large model. Nobody has published the curve that tells you where to put the
gate. This repository has the data to publish it. Track B builds that artifact.

Track A fixes four claims that are currently false or self-contradicted by the
repository's own receipts.

## Hard rules

1. Never run `git merge`, `gh pr merge`, or a rebase that lands one branch on
   another. Never push to `main` or `master`. Create a feature branch and push
   only that. Opening a pull request is fine. Pushing the branch is the stop
   point.
2. Never hand-write a metric into a Markdown, HTML, or JavaScript file. Every
   number a human reads is generated from a JSON receipt under `reports/v2/`
   through `scripts/render_receipts.py`. If you are typing a percentage, you are
   doing the wrong thing.
3. Never invent a number. If a script has not run, its receipt does not exist and
   the number goes nowhere. Report what commands actually printed, including
   failures.
4. Show real output. Do not summarise a test run as passing. Paste the tail.
5. Publish results that are worse than the current claims. Track B may show the
   cascade is expensive at realistic base rates. If so, publish that. Never tune
   a threshold, a cost assumption, or a sample until the number improves.
6. Keep changes surgical. Do not refactor adjacent code or reformat untouched
   files.
7. Leave `oss/`, `files-not-tobe-commited/`, and `RLCD Cookbook/` untouched. They
   contain third-party material and are already gitignored. Do not commit them.
8. Prose style: active voice, second person, present tense, sentence-case
   headings, Oxford comma. No em dashes. No "not X but Y" constructions. No
   bold-label bullets. No Summary or Conclusion section unless asked.

## Track A, in order

A1, remove the reinforcement learning claim. There is no RL in this repository
and the README claims there is. Rename the `rlcd/` package to `verdict/`, retitle
the README section, rename the HuggingFace repo to `verdict-modernbert-151m`
(HuggingFace keeps a redirect), and update `artifacts/ARTIFACTS.json` and every
badge. Keep one sentence saying the work is inspired by TypeSafe's described RLCD
approach and trained here with supervised proper scoring rules instead.

A2, set the shipped calibrator temperature to 1.0. The report shows uncalibrated
beats calibrated on NLL, Brier, and both ECE estimators, yet `worker.js:244`
divides by 1.4265 on every browser inference. Keep the fitted temperature in the
report as a measured result.

A3, correct the latency claim. The README says "under 35 milliseconds" three
times. The receipt says K=5 p50 is 35.58 ms single thread. Use the measured
number and pin thread count to 1.

A4, branch the browser precision. fp16 when WebGPU initialises, fp32 on the WASM
fallback, because fp16 on CPU measures 99.77 ms against fp32's 35.58 ms. Record
the chosen file in the receipt.

A5, delete the "Key topics and buzzword taxonomy" section, drop the phrase
"post-trained foundational decision model", add a NOTICE file, and state the
Apache 2.0 licenses of ModernBERT and GLiClass explicitly.

Stop after A5 and show: the full test run, `render_receipts.py --check`, and a
diff summary.

## Track B, in order

B1, extend the predictions dump in `scripts/evaluate.py` to include the full
`probabilities` vector and `candidate_ids` per row, then re-run evaluation. Write
`scripts/cascade_analysis.py` producing `reports/v2/cascade_analysis.json` with
the threshold sweep, error placement, base-rate resampling, and cost model
described in the plan. Render its tables into the README through the existing
marker mechanism.

B2, add a Cascade tab to `webgpu-demo/index.html` driven entirely by
`cascade_analysis.json`, with a threshold slider, live readouts, a plotted
selective-risk curve, and a base-rate selector. It must render before the model
weights finish downloading.

B3, reframe the live engine tab copy around the confidence gate.

Stop after B3 and show the cascade table plus a screenshot description of the new
tab.

B4 is a screen recording the human will make. Prepare for it: make sure the demo
loads cleanly from a cold cache, the threshold slider visibly flips the policy
action on a single decision, and removing the correct option produces an
abstention. Report anything in that sequence that does not work.

## Do not

Do not build a fraud-email demo or any demo outside banking intent. The checkpoint
is Banking77-tuned and experiment E9 already measured 48.07% on out-of-domain
decision tasks against Jev's 90.80%. Demoing out of domain will perform badly and
contradicts a published receipt.

Do not write launch copy, tweets, or thread drafts. That is Track C and it happens
after a human reviews these numbers.

## Done means

- `CI_REQUIRE_ARTIFACTS=1 pytest tests/ -q` passes with no skips.
- `python scripts/render_receipts.py --check` exits zero.
- `grep -rn "Reinforcement Learning\|under 35 milli\|Sub-35ms\|buzzword" README.md`
  returns nothing.
- `reports/v2/cascade_analysis.json` exists and the README renders its tables.
- The Cascade tab renders with the weights blocked in devtools.
- The feature branch is pushed and `main` is untouched.
