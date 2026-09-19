# Kickoff prompt

Paste the block below into the agent on the training laptop. It is complete as written.

```
Read AGENTS.md first, then plans/Claude/IMPLEMENTATION_PLAN.md sections 1, 2 and 7, then RUNBOOK.md.
Do not read any other planning document until those four are done.

You are continuing Verdict 2.0. The code is written and smoke tested. Your job is to run it, not to
redesign it. Everything you need is already in this repository, including the Laya and Kev reference
code under Laya/ and Kev/. Do not clone or download any repository. Downloading model weights from
Hugging Face is expected and fine.

Hardware: GTX 1660 Ti, 6 GB with about 5.5 GB usable, compute capability 7.5. bfloat16 and
FlashAttention-2 both require compute capability 8.0 and will fail. Use --fp16.

Execute in this order and stop at any gate that fails.

STEP 1. Environment.
    python -m venv .venv && source .venv/bin/activate
    pip install torch --index-url https://download.pytorch.org/whl/cu121
    pip install -r requirements-train.txt
    python -c "import torch;print(torch.cuda.get_device_capability(0))"
Expect (7, 5). Report what it prints.

STEP 2. Smoke test, about two minutes.
    python -m verdict2.train --backbone answerdotai/ModernBERT-base --epochs 2 --limit 60 --batch_size 4 --out artifacts/smoke
Gate: it completes and dev accuracy is above 0.269, which is uniform random on this benchmark.
If it fails, fix the environment and stop. Do not change the model code to make a smoke test pass.

STEP 3. Baseline run on the small backbone, about 30 to 45 minutes.
    python -m verdict2.train --backbone answerdotai/ModernBERT-base --epochs 8 --batch_size 8 --accum 2 --fp16 --out artifacts/verdict2-base
Gate: dev accuracy at or above 0.700. That is five points above the TF-IDF floor recorded in
reports/reference_floors.json. If it fails, the problem is the data path or the objective, not model
size. Diagnose it. Do not jump to a larger backbone to escape a failing gate.

STEP 4. Loss weight sweep on the small backbone, dev fold only.
Sweep --lambda_brier over 0.0, 0.25, 0.5, 1.0 and --perm_kl over 0.0, 0.5, 1.0.
Keep every result, including the ones that lose. Choose on dev accuracy and dev Brier together,
and record the chosen values.

STEP 5. Full run on the large backbone, about 1.5 to 2 hours.
    python -m verdict2.train --backbone answerdotai/ModernBERT-large --epochs 8 --batch_size 4 --accum 4 --fp16 --optim8bit --grad_checkpoint --out artifacts/verdict2-large
Carry the winning weights from step 4. The two extra flags are what keep this inside 5.5 GB: fp32
Adam states alone are 3.2 GB for a 396M model. If it still runs out of memory, set --batch_size 2
and --accum 8, which keeps the effective batch the same.
Gate: dev accuracy at or above 0.760 and dev Brier at or below 0.120.

STEP 6. One test read. Only after step 5 passes.
    python -m verdict2.evaluate --checkpoint artifacts/verdict2-large/model.pt --out reports/verdict2_test.json --confirm
Running it without --confirm prints an anti-leak checklist. Read that checklist and confirm each
item is actually true before passing the flag. Record how many times you read the test split.

STEP 7. Report. Write reports/verdict2_summary.md containing one table with all of these together:
accuracy, soft accuracy, Brier, ece_distribution, ece_confidence, score MAE, within one level, the
full permutation_stability block including its kev_reference numbers, and every row from
reports/reference_floors.json.
Never quote ece_confidence without ece_distribution beside it. One is the stock harness metric and
the other is the correctness head, and publishing only the flattering one is metric gaming.

Rules you must not break.
1. Never merge anything and never push to main on a public repository. Work on a branch and open a
   pull request. The repository owner merges.
2. The test split is a vault. Temperature, the correctness head, early stopping and every
   hyperparameter choice are fitted on the train split's calib and dev folds. Never fit on test.
3. Do not add an abstention or "none of the above" option. This benchmark has none, and injecting a
   dummy option corrupts the label space.
4. If a number you produce contradicts a fact in the "Facts that are measured" section of AGENTS.md,
   stop and report the contradiction. Do not quietly overwrite it.
5. Report what actually happened. If a gate fails, say so with the output. Do not round a failing
   number up to a passing one.

When you have finished, report: the metrics table, which gates passed and which failed, how many
test reads you made, total wall clock time, and anything you could not run and why.
```
