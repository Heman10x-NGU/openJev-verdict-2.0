# Launch reply kit

Not for the launch page. These are pre-drafted answers for when someone challenges a number in the
replies. Every one of these is a question a competent reader *will* ask — having the answer ready
turns a challenge into a second round of engagement instead of a retraction.

Rule of thumb: **concede the narrow point instantly, then restate the wide one.** You lose nothing
by agreeing that a 0.5-point gap is a 0.5-point gap, and you win the thread by pointing out it came
from a model a third the size.

---

### "Your 0.0144 ECE is a different channel than Laya's 0.214. Apples to oranges."

Correct, and both numbers are published side by side in the table.

On the channel both models expose — the distribution channel — we're at **0.1513 against Laya's
0.2140**, so we're 29% tighter there too. The 0.0144 is a *second* output Laya doesn't have, not a
substitute for the first. That's the whole design: one channel reports the panel, the other reports
whether we're right.

---

### "0.5 points of accuracy over Laya on 2,000 decisions is statistical noise."

Agreed on that line specifically — it's about 0.4 sigma, and we wouldn't defend "record" on
accuracy alone.

The claim is **parity with a 2.8× larger model**: 149.6M vs 421.3M. And on the two lines that
aren't within noise, we're ahead — Brier 0.0636 vs 0.0660, distribution ECE 0.1513 vs 0.2140.
Matching a 421M model at 149.6M is the result; the extra half point is a rounding bonus.

---

### "TF-IDF has a better ECE than you. 0.0207 vs 0.1513."

True, and it's in our own table — we published the floors rather than hiding them.

TF-IDF gets there by being diffuse: 66.10% accuracy and a Brier of 0.1520. It's well calibrated
about being wrong a third of the time. We're 11 points more accurate with a Brier 2.4× better, and
we ship a confidence channel at 0.0144. Low ECE on its own isn't a decision system.

---

### "Is the confidence head actually doing anything, or just predicting the base rate?"

The sharpest question in the set. Honest answer: **mean head confidence is 0.7626 against 77.10%
accuracy**, so a constant predictor would score well on ECE too. On the calibration fold the head's
BCE beats a base-rate constant by roughly 0.02–0.04 nats — real signal, but modest.

ECE measures calibration, not discrimination. We're publishing **AUROC and a risk–coverage curve**
next, which is the right way to measure whether the head separates right from wrong. Don't claim
"knows when it's wrong" until those land.

---

### "Did you tune on the test set?"

No, and it's checkable.

The 1,200 training cases were split **at the case-id level** into fit / calib / dev — sibling
questions from one case can't straddle a fold. Checkpoint selection used dev. Temperature and the
confidence head were fit on calib. The test split was read **once**, behind a `--confirm` checklist
gate in `verdict2/evaluate.py`. Verified: zero case-id overlap and zero duplicate state payloads
between train and test.

---

### "Jev's numbers are unverified."

Correct — marked with † on the page. It's a vendor-published figure and we did not re-measure it on
this harness. If someone wants to run real Jev through `verdict2/evaluate.py`, we'll publish
whatever comes back.

---

### "Kev was never evaluated on typed-decisions."

Correct — marked with ‡, and we say so in the table itself. Only Kev's **flip rate** is comparable,
because it's the same option-order perturbation. We don't claim an accuracy or Brier win over Kev,
and Kev's accuracy row is intentionally blank.

---

### "4.76% flips on what, exactly?"

Choice questions with 3 or more options — about **29% of the benchmark**, 2,918 perturbed
comparisons across 5 trials. Ordinal `score` questions are deliberately never permuted because
their order carries meaning; permuting them would be a bug, not a test.

And 4.76% isn't zero — roughly one decision in twenty-one still flips. It's 36% below Kev's
published 7.41%, not a solved problem.

---

### "Permutation-KL is what made it order-robust? Prove it."

We can't yet, and we should say so rather than bluff. The KL term contributed roughly 0.5% of total
loss and converged by epoch 3, which suggests the **single-pass bidirectional architecture** — all
options scored inside one attention pass — is doing most of the work.

The `--perm_kl 0` ablation is pre-registered in `RUNBOOK.md §5` and hasn't been run. It's the next
experiment. If the flip rate barely moves without the term, that's a *more* interesting finding than
the one we'd have claimed.

---

### "Your temperature table has gaps."

Yes — 5 of 24 `(question type, cardinality)` buckets were fitted, because those are the only five
shapes present in the calibration fold. The test split contains the same five, so nothing was
uncalibrated in the reported run.

A shape outside those buckets falls back to temperature 1.0 silently. An assertion plus a
nearest-bucket fallback is going in before anyone runs this in production.

---

### "Why didn't you train ModernBERT-large?"

The gate was pre-registered in `RUNBOOK.md §4` *before* training started: dev accuracy ≥ 0.760 and
dev Brier ≤ 0.120. Base hit **0.785 / 0.0639** — it cleared the bar set for the large model.

Base also converged: dev accuracy over the final four epochs was 0.7775 → 0.7850 → 0.7762 → 0.7825.
Large needs 8-bit AdamW and gradient checkpointing to fit in 6 GB at all, and would run 14–18 hours
on this card. We're not claiming Large wouldn't help — we're saying Base already cleared the gate.

---

### "Trained on a laptop" — is that real?

GTX 1660 Ti, 6 GB, Turing TU116, **no tensor cores**. 8 epochs, 31,757 seconds wall clock, ~5.8 GB
peak VRAM. Best checkpoint at epoch 6. The full epoch-by-epoch log is in the repo, and `RUNBOOK.md`
has the exact command.
