# Cold Email: TypeSafe AI (Founders & Engineering Team)

Target recipients:
- **Diogo Almeida** (Co-founder & CEO, former OpenAI researcher, co-inventor of RLHF)
- **Erik Gafni** (Co-founder & CTO)

Style rules applied:
- Zero AI buzzwords (no "thrilled", "revolutionize", "game-changer", "honestly", "delve").
- Zero em-dashes.
- Zero negative parallelisms ("not X, but Y").
- Direct, active voice, technical specifics only.
- Concrete numbers and verifiable repository receipts.

---

## Option 1: Tailored for Diogo Almeida (CEO)
*Subject:* Verdict 2.0: 150M ModernBERT decision engine hitting 1.44% ECE on typed-decisions

Hi Diogo,

I followed TypeSafe AI's launch of Jev out of stealth this week. Building fast, deterministic probabilistic decisions rather than autoregressive text generation is the right architecture for enterprise automation.

Inspired by Jev, I built an open-source reproduction called Verdict 2.0 using answerdotai/ModernBERT-base (149.6M parameters). I trained it on a consumer laptop GPU in 8.8 hours and evaluated it on the LocalLLaMA/typed-decisions benchmark (2,000 held-out test decisions).

On that benchmark, Verdict 2.0 reaches 77.10% top-1 accuracy, matching and outperforming Laya's 421M model (76.60%) and Jev 1.13.0 (72.70%).

The core problem I solved was the soft-label calibration trap. Because teacher panel labels have human doubt (mean top probability of 0.659), any single probability distribution cannot simultaneously match the soft distribution (low Brier) and track empirical correctness (low ECE).

Verdict 2.0 decouples inference into two distinct channels:
1. Distribution channel: Marker-pointer logits trained on soft cross-entropy and per-bucket temperature scaling (Brier 0.0636, ECE 0.1513).
2. Correctness channel: An MLP head trained out-of-fold on calibration predictions (ECE 0.0144, AUROC 0.7861).

This dual-channel design yields a 15x lower calibration error than Laya (0.2140) and a 10x lower error than Jev (0.1440). In selective routing, filtering the bottom 20% of uncertainty raises accuracy to 85.00%, and filtering the bottom 40% raises accuracy to 90.21%.

Because inference batches all workflow questions in one forward pass, latency is 25 ms per case, compared to Jev's 710 ms and Laya's 156 ms. The weights fit in under 600 MB of VRAM, making WebGPU execution viable.

The model weights, reproduction scripts, and test receipts are available here:
[GitHub Repository Link: https://github.com/Heman10x-NGU/Verdict-2.0-open-jev]

I would welcome your feedback on this dual-channel formulation. If you are looking for engineers who can build and optimize small, calibrated decision infrastructure, I would like to speak with you.

Best regards,

[Your Name]
[Your Phone Number]
[Your GitHub / LinkedIn URL]

---

## Option 2: Tailored for Erik Gafni (CTO)
*Subject:* 25ms single-pass decision inference and 1.44% ECE on ModernBERT-base (Verdict 2.0)

Hi Erik,

I saw TypeSafe AI's release of Jev. I have been working on the same problem space: evaluating typed decision primitives (Choice, Score, Noul) in non-autoregressive forward passes.

I recently finished training Verdict 2.0, a 149.6M parameter ModernBERT-base engine evaluated on the LocalLLaMA/typed-decisions test split (2,000 decisions).

Here is the engineering setup and measured results:

1. Single-pass batching: Instead of re-encoding the workflow state for every question, Verdict 2.0 encodes the state and evaluates all 5 questions in a single forward pass. Latency is 25 ms per 5-question case, compared to 710 ms for Jev 1.13.0 and 156 ms for Laya.

2. Symmetric Permutation-KL: To resolve prompt option-order bias, I trained twin passes under permuted option orders with a symmetric KL penalty. On the Kev evaluation protocol across 2,918 perturbed decisions, the argmax flip rate dropped to 4.76% (36% fewer flips than Kev-0.5B's 7.41%), with a p90 probability spread of 0.0915 (vs Kev's 0.2486).

3. Decoupled calibration head: Soft teacher distributions make binned ECE conflict with Brier scores. I added an out-of-fold CorrectnessHead over distribution shape. It achieves 0.0144 ECE (1.44% calibration error) and 0.7861 AUROC.

4. Top-1 exact accuracy: 77.10%, beating Laya (421M Large, 76.60%) and Jev 1.13.0 (72.70%).

5. Footprint: Under 600 MB unquantized, suitable for edge and in-browser WebGPU execution.

The training pipeline, audited JSON test receipts, and model checkpoints are available here:
[GitHub Repository Link: https://github.com/Heman10x-NGU/Verdict-2.0-open-jev]

I am interested in joining TypeSafe AI to work on low-latency, calibrated decision infrastructure. Are you open to a brief technical discussion this week?

Best regards,

[Your Name]
[Your Phone Number]
[Your GitHub / LinkedIn URL]

---

## Option 3: Short Direct Message (Twitter / X or LinkedIn)

Hi Diogo,

I followed TypeSafe AI's launch of Jev. Inspired by your architecture, I trained Verdict 2.0, a 149.6M ModernBERT decision engine evaluated on LocalLLaMA/typed-decisions.

Results on 2,000 held-out decisions:
- 77.10% top-1 accuracy (beats Laya 421M's 76.60% and Jev's 72.70%)
- 25 ms latency (single-pass forward pass, 28x faster than Jev)
- 1.44% ECE via a decoupled correctness head (15x tighter calibration than Laya)
- 4.76% option flip rate under permutations (vs Kev's 7.41%)
- Sub-600 MB VRAM footprint (WebGPU ready)

Code, weights, and audited receipts are open here: [Repository Link]

I would love to contribute to TypeSafe AI's engineering team. Are you hiring engineers in this space?
