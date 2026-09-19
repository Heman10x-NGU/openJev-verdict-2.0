# Launch post, corrected

Every number below is from `reports/verdict2_base_test.json`. The claims audit is in
`docs/VERIFIED-CLAIMS.md`. Two figures in the earlier draft were wrong and are fixed here: latency
was reading a throughput number as milliseconds, and the accuracy comparison against Laya does not
survive a significance test.

Before posting, run `scripts/analysis/selective_classification.py` and replace the AUROC line with
whatever it prints. If you cannot run it in time, delete that bullet.

```
A few months ago I shipped Verdict 1.0 and it scored 26.10% on LocalLLaMA/typed-decisions.

Uniform guessing scores 26.9%. It was worse than random.

Instead of quietly deleting it, I went looking for why. Being reliably worse than chance takes
information, so it was not a weak model, it was a broken one.

I rebuilt it as Verdict 2.0. Receipts below, all on the same 2,000 held-out enterprise decisions.

1. Accuracy 77.10%

That is a 149.6M encoder matching Laya's 421M ModernBERT-large (76.60%) with 2.8x fewer parameters.
I am not claiming a win there: the 95% CI is [75.3%, 78.9%] and Laya sits inside it. Parity at a
third of the size is the result.

It does clear TypeSafe Jev's reported 72.70% and the benchmark's own teacher-agreement ceiling of
73.5%, both outside the interval.

2. Calibration, which is where it actually wins

Like for like, expected calibration error on the top probability: 15.13% against Laya's 21.40%.
29% lower.

The reason this is hard: the benchmark scores Brier against a soft reviewer-panel distribution and
ECE against hard correctness. Those pull in opposite directions. Match the panel and your ECE
explodes. Sharpen until confidence tracks accuracy and your Brier explodes. Laya took the Brier
side and pays 21.40% ECE for it.

So I stopped asking one probability vector to answer both questions. Verdict 2.0 reports two
channels: a distribution channel that estimates what the panel would say (Brier 0.0636), and a
separate correctness head that estimates whether this particular answer is right (ECE 1.44%).

3. Option order barely moves it

Shuffle the candidate list and Kev-0.5B changes its answer 7.41% of the time on their fixture.
Verdict 2.0 flips 4.76%, with a p90 probability spread of 0.0915 against their 0.2486. 36% fewer
flips, 63% tighter spread. Trained with a symmetric KL between two orderings of the same question.

4. 202 ms per five-question case, against Jev's reported 710 ms

3.5x faster, measured at 24.7 decisions per second on one GPU. On CPU through ONNX it is 35.6 ms
per question single-threaded.

5. Trained on a 6 GB laptop GPU

GTX 1660 Ti. No bf16, no FlashAttention-2, both need a newer card. Full fine-tune fits with 8-bit
Adam.

6. Discipline, because the numbers are only worth what the protocol is worth

Splits hashed at the case level so sibling questions never leak across folds. Temperature and the
correctness head fitted only on a calibration fold. The test split gated behind a checklist in the
eval script and read once.

The honest part: Jev's 72.70% is a figure I inherited from a community notebook, not something I
measured. Somebody should run the real Jev on this harness. I would rather that number be verified
than flattering.

Weights, receipts, and replication code:
GitHub: https://github.com/Heman10x-NGU/openJev-verdict-2.0
Hugging Face: https://huggingface.co/heman10x/openJev-verdict-2.0
Dashboard: https://heman10x-ngu.github.io/openJev-verdict-2.0/

Critiques welcome, especially on the calibration split.
```

## Why the changes

The original claimed 25 ms per case and 28x faster. The receipt says 24.7 decisions per second,
which is 202 ms per five-question case and 3.5x. That is the easiest claim in the post to check,
because the JSON is public.

The original claimed a win over Laya on 0.5 points. With n=2000 the standard error is 0.94 points
and p is 0.71. Leading with "matches at 2.8x fewer parameters" is both true and a better hook.

Naming the unverified Jev number yourself is cheaper than having a reply do it, and it reads as
confidence rather than hedging.
