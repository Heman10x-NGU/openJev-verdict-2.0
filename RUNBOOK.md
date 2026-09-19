# Verdict 2.0 runbook

Everything needed to train and evaluate Verdict 2.0 on the GTX 1660 Ti laptop. The plan and the
reasoning behind it are in `plans/Claude/`. This file is the operational sequence.

Target hardware: GTX 1660 Ti (6 GB, about 5.5 GB usable), Ryzen 7 4800H, 16 GB RAM.

## 0. Hardware constraints that will bite you

The 1660 Ti is Turing TU116, compute capability 7.5. Three things do not work on it:

- bfloat16 needs compute capability 8.0. Use `--fp16`. Any config with `bf16=True` fails.
- FlashAttention-2 needs compute capability 8.0. The code uses PyTorch SDPA, which works on 7.5.
- TU116 has no tensor cores, unlike the RTX 20 series. fp16 still runs at roughly twice the fp32
  rate, so `--fp16` is worth using, but do not expect tensor-core speedups.

Memory arithmetic that decides the flags, for full fine-tuning:

| Backbone | Params | fp32 params + grads | AdamW fp32 states | Total | Fits 5.5 GB |
|---|---|---|---|---|---|
| ModernBERT-base | 149.6M | 1.2 GB | 1.2 GB | about 2.8 GB | yes, plain AdamW |
| ModernBERT-large | 395.9M | 3.2 GB | 3.2 GB | about 6.8 GB | no |
| ModernBERT-large | 395.9M | 3.2 GB | 0.8 GB (8-bit) | about 4.5 GB | yes, with `--optim8bit` |

So: base trains with default flags, large needs `--optim8bit --grad_checkpoint`.

## 1. Setup

```bash
git clone https://github.com/Heman10x-NGU/Verdict-2.0-open-jev.git
cd Verdict-2.0-open-jev
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate

pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements-train.txt
```

Confirm the GPU is visible and is what you expect:

```bash
python -c "import torch;print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
```

Compute capability should print `(7, 5)`. If it prints `(8, x)` or higher you are on a different
card and can drop `--optim8bit`.

## 2. Smoke test first, always

Two minutes, and it catches every environment problem before you commit to a long run:

```bash
python -m verdict2.train --backbone answerdotai/ModernBERT-base \
  --epochs 2 --limit 60 --batch_size 4 --out artifacts/smoke
```

Expect dev accuracy well above 0.269 (uniform random) and a finite loss. If this fails, stop and
fix the environment. Do not proceed.

## 3. Develop on base

Fast iteration loop, roughly 30 to 45 minutes:

```bash
python -m verdict2.train --backbone answerdotai/ModernBERT-base \
  --epochs 8 --batch_size 8 --accum 2 --fp16 \
  --out artifacts/verdict2-base
```

Gate before moving on: dev accuracy at or above 0.700. That is five points above the TF-IDF floor in
`reports/reference_floors.json`. If base cannot clear 0.700, the problem is the data path or the
objective, not model capacity, and a bigger backbone will not rescue it.

## 4. Ship on large

Roughly 1.5 to 2 hours. The two extra flags are what keep it inside 5.5 GB:

```bash
python -m verdict2.train --backbone answerdotai/ModernBERT-large \
  --epochs 8 --batch_size 4 --accum 4 --fp16 --optim8bit --grad_checkpoint \
  --out artifacts/verdict2-large
```

If it still runs out of memory, drop `--batch_size` to 2 and raise `--accum` to 8. The effective
batch stays the same, so results should not move.

Gate before touching the test split: dev accuracy at or above 0.760 and dev Brier at or below 0.120.

## 4b. Order invariance, the Kev comparison

`--perm_kl` adds a symmetric KL between the same question scored under two option orderings, which
is Kev's technique. Kev still records a 7.41% argmax flip rate with it. `verdict2/evaluate.py`
reports the same three numbers Kev publishes (argmax flip rate, mean probability spread, p90 spread)
with Kev's values inlined as `kev_reference`, so the comparison needs no external lookup.

It defaults to 0.5. Sweep it in step 5 and keep whichever value wins on dev.

## 5. Sweep the loss weights, on dev only

The objective mixes soft cross-entropy, a Brier term, and a small hard-label term. The starting
weights are a guess, so settle them with measurements:

```bash
for b in 0.0 0.25 0.5 1.0; do
  python -m verdict2.train --backbone answerdotai/ModernBERT-base --epochs 6 \
    --lambda_brier $b --out artifacts/sweep-brier-$b
done

for k in 0.0 0.5 1.0; do
  python -m verdict2.train --backbone answerdotai/ModernBERT-base --epochs 6 \
    --perm_kl $k --out artifacts/sweep-permkl-$k
done
```

Pick the winner on dev accuracy and dev Brier together. Record the losers too; a negative result is
worth writing down.

## 6. Single test read

The test split is a vault. Read it once per candidate, after everything else is frozen:

```bash
python -m verdict2.evaluate --checkpoint artifacts/verdict2-large/model.pt \
  --out reports/verdict2_test.json --confirm
```

Running it without `--confirm` prints the anti-leak checklist and exits. Read the checklist honestly
before passing the flag. Every read you make is a read you have to report.

## 7. What to report

Publish these together, never a subset:

- accuracy, soft accuracy, Brier, score MAE, within-one-level
- both calibration numbers: `ece_distribution` (the stock harness metric, on `max p`) and
  `ece_confidence` (the correctness head). Quoting only the second reads as metric gaming.
- the full permutation stability block, which carries Kev's numbers inline for comparison
- the reference floors from `reports/reference_floors.json` in the same table

The floors matter because a TF-IDF baseline on this benchmark already reaches about 0.65 accuracy
with an ECE near 0.03. Any headline number has to be read against that.

## 8. Phase 0, run this on the Mac

The Verdict 1.0 checkpoint lives in `artifacts/v2` on the Mac and is too large for git. Run the
diagnosis there, not on the training laptop:

```bash
python scripts/analysis/phase0_diagnose.py --limit 120
```

Verdict 1.0 scores 0.2610 where uniform guessing scores 0.2690, with a Brier of 0.5851 against
uniform's 0.2433. Being reliably worse than chance takes information, which points at a wiring
defect rather than a weak model. The script perturbs option order and the id/description mapping and
reports whether accuracy moves. If it does, find the defect before reusing any of that code path.

## Command reference

| Command | Purpose |
|---|---|
| `python -m verdict2.train --help` | All training flags |
| `python -m verdict2.evaluate --help` | Final evaluation flags |
| `python scripts/analysis/reference_floors.py` | Regenerate the baseline floors |
| `python scripts/analysis/phase0_diagnose.py` | Diagnose the v1 baseline (Mac) |
