# Launch and marketing plan

Two workstreams. Track A is the four blockers, roughly two hours, and must finish
before anything is posted. Track B is the demo that earns the attention.

The positioning: everyone in this discussion is adopting the confidence-cascade
pattern (fast calibrated model, confidence gate, escalate the rest). Nobody has
published the curve that tells you where to set the gate. You have it.

---

## Track A: blockers

### A1. Remove the reinforcement learning claim
There is no RL in this repository. The loss is `CE + 1.0 * Brier` optimised with
AdamW. Grep for reward, advantage, rollout, PPO, REINFORCE, policy gradient
returns nothing.

- Rename the `rlcd/` package to `verdict/`.
- Retitle the README section "Reinforcement Learning for Calibrated Decisions
  (RLCD)" to "Calibrated decision training".
- Rename the HuggingFace repo from `rlcd-modernbert-151m` to
  `verdict-modernbert-151m`. HuggingFace supports rename with a redirect, so old
  links keep working. Update `artifacts/ARTIFACTS.json` repo_id and all badges.
- Keep one honest sentence: inspired by TypeSafe's described RLCD approach,
  trained here with supervised proper scoring rules rather than RL.

### A2. Ship temperature 1.0
`evaluation_report_v2.json` shows uncalibrated beats calibrated on every metric:
NLL 0.1731 against 0.1768, Brier 0.0756 against 0.0785, ECE equal-width 1.13%
against 3.35%, ECE adaptive 0.83% against 2.87%. `worker.js:244` divides logits
by 1.4265 on every browser inference, degrading the live demo.

Set the shipped calibrator to T=1.0. Keep the fitted value in the report as a
measured result. The claim becomes stronger: the composite Brier objective
produced a natively calibrated model that needs no post-hoc correction.

### A3. Correct the latency claim
`exp_e7_latency.json` fp32 single thread: K=5 p50 is 35.58 ms, K=25 is 140.10 ms.
The README says "under 35 milliseconds" in three places. Replace with the
measured p50 at K=5 and link the K sweep. Pin thread count to 1, because 4-thread
is slower at K=5 (49.47 ms against 35.58 ms).

### A4. Fix the browser precision choice
fp16 single thread K=5 is 99.77 ms against fp32's 35.58 ms, because ONNX Runtime
CPU has no native fp16 kernels and inserts cast operations. `worker.js:122`
loads fp16 unconditionally.

Branch on the provider: fp16 when WebGPU initialises, fp32 on the WASM fallback.
Record the chosen file in the receipt.

### A5. Cosmetic, same pass
- Delete the "Key topics and buzzword taxonomy" section. Visible keyword
  stuffing, discounted by search engines, and this audience screenshots it.
- Drop "post-trained foundational decision model". It is a 151M GLiClass
  fine-tune.
- Add a `NOTICE` file and state the Apache 2.0 license of ModernBERT and
  GLiClass explicitly in the credits section.

---

## Track B: the cascade demo

### B1. `scripts/cascade_analysis.py` (the headline artifact)

Reads `reports/v2/predictions_v2.jsonl` and produces the economics of a
confidence-gated cascade at every threshold. No API calls, no new inference.

For each threshold in 0.50 to 0.99 step 0.01, report: escalation rate, auto
handled share, errors retained above the gate, errors caught below it, retained
accuracy on the auto-handled set with a Wilson interval, and end-to-end accuracy
assuming the escalation tier is correct at a configurable rate (default 0.95,
swept over 0.85 to 1.00).

Three additional views, each answering a specific unanswered question from the
thread:

1. **Error placement.** Of all errors, how many sit above the gate and execute
   silently, and how many fall below it. Current measured split at 0.95 is 2
   above and 48 below out of 50.
2. **Base-rate shift.** Resample the test set to abstention or positive-class
   prevalences of 1%, 5%, 10%, 25%, 50% and recompute the escalation rate at each
   threshold. This answers whether a gate tuned on a balanced set still sends a
   third of traffic upstream on a realistic 1% stream.
3. **Cost model.** Take a local cost per decision and an escalation cost per
   decision as arguments, and emit total cost per 1,000 items at every threshold,
   with the accuracy achieved. Default the escalation cost to a published
   frontier price so the table is legible, and label it as an assumption.

Write `reports/v2/cascade_analysis.json` and render a markdown table into the
README through the existing `render_receipts.py` mechanism.

Note: `predictions_v2.jsonl` currently stores only the top-1 confidence. The
base-rate resampling needs the full probability vector per row. Extend the
predictions dump to include `probabilities` and `candidate_ids`, then re-run
`scripts/evaluate.py`. This is the only re-run Track B needs.

### B2. Cascade tab in the browser demo

Add a fourth tab beside Live Engine, Failure Gallery, and Audit Receipts.

- A threshold slider from 0.50 to 0.99.
- Live readout of escalation rate, auto-handled share, retained accuracy, and
  errors above the gate, all read from `cascade_analysis.json`.
- A plotted selective-risk curve with the current threshold marked.
- A base-rate selector showing how the escalation rate moves.

This is static data rendered client side. It needs no model inference, so it
loads instantly even before the 300 MB weights arrive, which also fixes the
current dead air during model download.

### B3. Make the existing playground demo the right story

The live engine tab already lets a visitor edit candidates and remove the correct
answer. Reframe the copy around the gate: set a threshold, watch a decision land
above or below it, and see the policy action change. The receipt download already
carries the threshold and policy action.

### B4. Screen recording

Record one take, 45 to 70 seconds, no narration, captions only. Sequence:

1. Page loads, WebGPU badge lights, weights stream in with the byte counter.
2. Paste a banking query, run it, decision appears with the probability bars and
   the millisecond timing.
3. Drag the threshold slider down and up, and show the policy action flipping
   between execute and escalate on the same decision.
4. Delete the correct option from the candidate list, re-run, model abstains.
5. Cut to the cascade tab, drag the threshold, show escalation rate against
   retained accuracy moving together.
6. End on the errors-above-gate number.

Do not speed the video up. Hassan's post explicitly says his video is not sped up
and people noticed. Record at the real 36 ms and let it be fast on its own.

### B5. What not to demo

Do not build a fraud-email demo. The checkpoint is Banking77-tuned and E9 already
measured 48.07% on out-of-domain decision tasks against Jev's 90.80%. A fraud demo
would either need new fine-tuning or would perform badly on camera. Demo the
domain the model is actually good at, and let E9 stand as the published honest
limit.

---

## Track C: the post

Write after Track A and B land. Constraints: under 280 characters per post, no em
dashes, precision beside recall, no latency figure without a percentile.

The thread shape:

1. The cascade curve, as the answer to a question the timeline is actively asking.
2. The order-invariance result: 3.0% flip rate against 27.8% for a 4B decoder on
   the same perturbation.
3. The shuffled-context control: 95.0% to 22.4%, abstention 21.8% to 72.7%.
4. The honest limit: E9, 48.07% against Jev's 90.80%, and E5, abstention recall
   collapsing from 75.5% to 10.0% under hard negatives plus paraphrase.
5. Repo, weights, browser demo.

Reply to Hassan's post with the error-placement number and a link to the curve.
Do not claim to beat him. Different task, different data. Offer the tool.

---

## Sequencing

Track A first and completely. Then B1, which is the artifact everything else
displays. Then B2 and B3. Then record B4. Then write Track C against final
numbers.
