You are working in a public open-source repository at:

/Users/heman10x/Downloads/claude_dev/personal_projects/Mind-Palace/RLCD-demo

Read `Plans/GENERALIZATION-EXPERIMENTS-PLAN.md` first. It is the specification.
Build experiments G1 through G4 in that order. Do not start Tier 2.

## What this repository is

A 151M ModernBERT plus GLiClass encoder fine-tuned on PolyAI Banking77 to answer
typed decision questions in one forward pass. Given text plus a list of candidate
option descriptions, it returns one calibrated probability per candidate plus an
explicit abstention option. The classification head produces 25 logits, so a
question supports at most 24 substantive candidates plus abstention.

Two checkpoints matter:
- Base: `knowledgator/gliclass-modern-base-v2.0`, already in the HuggingFace
  cache. A general zero-shot classifier.
- Fine-tuned: `artifacts/v2/model.safetensors`, the shipped Banking77 model.

## The hypothesis you are testing

A 6-example probe on authored phishing emails scored the base checkpoint 5/6 and
the fine-tuned checkpoint 1/6. The fine-tuned model abstained on 5 of 6 at
confidence 0.58 to 0.80. It refused rather than guessed wrong.

The hypothesis is that Banking77 fine-tuning taught a domain gate ("this is not a
banking intent, so abstain") rather than a transferable decision skill, and that
this explains the existing experiment E9, where the same checkpoint scored 48.07%
on out-of-domain decision tasks.

Six authored examples is not evidence. Your job is to test this properly on
public data at scale, across four task families, and report what comes out.

**Run both checkpoints on every experiment and report both columns.** That
comparison is the deliverable. An experiment that reports only one checkpoint is
incomplete and must be redone.

## Hard rules

1. Never run `git merge`, `gh pr merge`, or a rebase that lands one branch on
   another. Never push to `main` or `master`. Create a feature branch and push
   only that. Pushing the branch is the stop point.
2. Never hand-write a metric into a Markdown, HTML, or JavaScript file. Every
   number a human reads comes from a JSON receipt under `reports/v2/` rendered
   through `scripts/render_receipts.py`. If you are typing a percentage, you are
   doing the wrong thing.
3. Never invent a number. If a script has not run, its receipt does not exist and
   the number goes nowhere. Report what commands actually printed, including
   failures and crashes.
4. Show real output. Do not summarise a run as passing. Paste the tail.
5. **Never tune anything after seeing a result.** Do not adjust a prompt, a
   threshold, a label description, or a sample to improve a number. If you want
   to test a different label wording, it is a separate recorded arm and every arm
   gets reported. Changing a setting because the number was disappointing is
   fabrication.
6. The expected outcome is that the fine-tuned checkpoint loses to the base
   checkpoint on these tasks. That is a publishable finding, not a failure to
   engineer around. Report it plainly.
7. Commit the exact dataset row ids you sampled, plus the seed, so every number
   is reproducible.
8. Keep changes surgical. Do not refactor adjacent code or reformat untouched
   files. Do not regress the existing 42 tests or
   `scripts/render_receipts.py --check`.
9. Leave `oss/`, `files-not-tobe-commited/`, and `RLCD Cookbook/` untouched and
   uncommitted.
10. Prose style: active voice, second person, present tense, sentence-case
    headings, Oxford comma. No em dashes. No "not X but Y" constructions. No
    bold-label bullets. No Summary section unless asked.

## Datasets, all verified reachable

- `ealvaradob/phishing-dataset` and `SetFit/enron_spam` for G1
- `TrustAIRLab/in-the-wild-jailbreak-prompts` and `tatsu-lab/alpaca` for G2
- `https://raw.githubusercontent.com/Shopify/product-taxonomy/main/dist/en/categories.txt`
  for G3
- G4 is a fixture you author, with a provenance manifest marking it
  project-authored

## Shared protocol for every experiment

At least 500 items per task with a committed seed. For each checkpoint report
accuracy, NLL, Brier, ECE equal-width and adaptive, abstention rate, abstention
precision and recall, and 1000-sample bootstrap 95% CIs on accuracy and
abstention rate. Report the selective-risk curve across thresholds 0.50 to 0.99.
Report the over-abstention rate, meaning the share of items where the model
abstained although the correct option was present, because the probe suggests
that is the whole story.

Reuse the existing machinery: `core/formatting.py::format_prompt` for prompts and
`core/calibration.py::compute_ece` for binning. Do not write new prompt
construction.

## Order and checkpoints

Build G1 first and stop. Show both checkpoints' full metric tables, the
over-abstention rates, and the cascade curve. G1 includes two label-framing arms,
banking-framed and neutral-framed, and both get reported. If the fine-tune
recovers under banking framing, say so explicitly, because that is direct
evidence for the domain-gate hypothesis.

Then G2, stop and report. Then G3, stop and report. Then G4.

G3 needs new code, `core/hierarchy.py`, with greedy and beam search over a tree
whose nodes hold at most 24 children plus abstention. Report leaf accuracy,
per-depth accuracy, forward passes per document, wall-clock per document, and
where errors concentrate, for greedy and for beam K=3 and K=5.

## Do not

Do not write launch copy, tweets, or thread drafts.

Do not showcase the `Score` or `Noul` primitives as capabilities. The repository
documents them as experimental and unbenchmarked. G2 renders its binary questions
through the existing 3-way Choice path in `core/formatting.py`, which is correct.

Do not delete, rename, or modify any existing receipt under `reports/v2/`.

## Done means

- `reports/v2/gen_g1_*.json` through `gen_g4_*.json` exist, each with both
  checkpoint columns.
- `CI_REQUIRE_ARTIFACTS=1 pytest tests/ -q` passes with no skips.
- `python scripts/render_receipts.py --check` exits zero.
- Sampled row ids and seeds are committed.
- The feature branch is pushed and `main` is untouched.
