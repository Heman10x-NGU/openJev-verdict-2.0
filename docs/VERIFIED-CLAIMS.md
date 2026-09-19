# Claims audit

Every number in the launch post, checked against the receipts in this repository.
Audited 2026-09-19 against `reports/verdict2_base_test.json`.

## Verified, safe to publish

| Claim | Receipt | Value |
|---|---|---|
| Top-1 accuracy 77.10% | `reports/verdict2_base_test.json` | 0.7710 over 2,000 decisions |
| Brier 0.0636 | same | 0.0636, which is 3.6% below Laya's 0.066 |
| Correctness-head ECE 1.44% | same | `ece_confidence` 0.0144 |
| Distribution ECE 15.13% | same | `ece_distribution` 0.1513, which is 29% below Laya's 0.214 |
| Option flip rate 4.76% | same | 0.0476 against Kev's published 0.0741, so 36% fewer |
| Probability spread 63% tighter | same | p90 spread 0.0915 against Kev's 0.2486 |
| Score MAE 0.2409, within one level 99.0% | same | beats Laya's 0.242 |
| Beats the teacher self-agreement ceiling | ceiling is 0.735, our 95% CI starts at 0.753 | separable |
| Per-workflow accuracy 73.2% to 81.2% | same | four workflows, 500 decisions each |

## Wrong. Fix before posting.

### 1. Latency is 202 ms per case, not 25 ms

The receipt records `throughput_decisions_per_second: 24.7`. That is decisions per second, not
milliseconds per case. The arithmetic:

```
24.7 decisions/second  ->  40.5 ms per decision  ->  202 ms per 5-question case
```

So the honest comparison against Jev's 710 ms is **3.5x faster**, not 28x. The 25 ms figure reads
the throughput number as if it were a duration, and it is the single most checkable claim in the
post. Anyone can open the JSON.

There is a separate, real latency receipt in `reports/v2/exp_e7_latency.json`: 35.6 ms p50 per
question at K=5, single-threaded ONNX on CPU. That is per question, so about 178 ms for a
five-question case on CPU. Quote that if you want a CPU number, and label it per question.

### 2. Accuracy does not separate from Laya

```
Verdict 2.0  0.7710   n = 2000   SE = 0.94 pp   95% CI [0.7526, 0.7894]
Laya         0.7660   inside that interval
difference   +0.5 pp  z = 0.37   p = 0.71
```

Laya sits inside our confidence interval, so "beats Laya's 76.60%" is not supportable and a
reviewer will say so in the first reply. Jev at 0.727 and the teacher ceiling at 0.735 are both
outside the interval, so those two comparisons hold.

The stronger and true claim: a 149.6M encoder **matches** a 421M one, with 2.8x fewer parameters,
while being better calibrated. Parity at a third of the size is a better story than a 0.5 point win
that does not survive a significance test.

### 3. The architecture claim is not what the code does

The post says Laya "re-encodes once per question" while Verdict 2.0 "batches all workflow questions
in a single forward pass". `verdict2/data.py` builds one sequence per question, each carrying the
full state, exactly as Laya does. The five questions are batched, not fused into one pass.

The latency advantage comes from the smaller backbone and from batching, not from a shared state
prefix. A shared-prefix engine is a future change, not a shipped one.

### 4. AUROC 0.7861 and the coverage numbers have no receipt

`grep -ri auroc` finds hits only in `README.md`, `docs/index.html`, and a comment in
`scripts/generate_laya_showdown_card.py`. No JSON computes it, and `verdict2/evaluate.py` calculates
neither AUROC nor a selective-classification curve.

The claims may well be true. Right now they are unbacked, in a post whose framing is "here are the
verified receipts." Run `scripts/analysis/selective_classification.py` (added in this branch) to
produce `reports/verdict2_selective.json`, then quote the numbers it prints. If they differ from
0.7861, 85.00% and 90.21%, use what it prints.

### 5. The calibration headline compares two different quantities

"Lower Calibration Error (1.44% ECE). Laya sits at 21.40%" puts our correctness-head channel next to
Laya's distribution ECE. Those measure different things.

Like for like, both on `max p`: ours 15.13%, Laya 21.40%, so 29% lower. That is a real win, keep it.
Note that Jev's published 14.40% is slightly **lower** than our 15.13%, so "beats Jev on
calibration" is false on that channel. Our correctness head at 1.44% is a genuinely separate and
stronger result, but it has to be labelled as a second channel rather than swapped in silently.

### 6. Two claims with no artifact

The "token-wiring defect where option markers were misaligned with the state payload" has no
`reports/phase0_diagnosis.json` in the repository. `scripts/analysis/phase0_diagnose.py` exists but
its output was never committed. Either commit the output or soften the claim to a description of
what you changed.

"Trained in 8.8 hours" has no duration receipt either. `train.py` prints total wall time; paste that
line into the repo if you want to keep the number.

## Numbers used for competitors

Laya's 0.766 accuracy, 0.066 Brier, and 0.214 ECE come from its published model card at
`convaiinnovations/laya-typed-decisions`, so they are citable.

Jev's 0.727 accuracy, 0.148 Brier, 0.144 ECE, and 710 ms are a hardcoded Python literal in a Laya
notebook. Nobody has measured Jev on this benchmark. Label that row as a vendor-reported figure, or
somebody else will.

Kev's 0.0741 flip rate is real, from its released `eval.json`, but Kev measured it on its own
fixture rather than on typed-decisions. Say "on their fixture" when you quote it.
