# Model choice, competitor status, strategy, and cost

Companion to `IMPLEMENTATION_PLAN.md` and `EXPLANATION.md`. Answers the five questions asked on
2026-09-19. Data comes from the Hugging Face MCP server (live Hub queries), the vendored `Laya/`
and `Kev/` trees, and the local dataset.

## 1. Hugging Face MCP status

Added and connected:

```bash
claude mcp add --transport http huggingface https://huggingface.co/mcp --scope user
```

It reports `huggingface: https://huggingface.co/mcp (HTTP) - Connected` and exposes four tools:
`hf_whoami`, `hub_repo_search`, `hub_repo_details`, and `hf_fs`. The tools register in a Claude Code
session at startup, so restart the session to call them directly. Every Hub figure in this document
came from that server, queried over HTTP in the meantime.

The Antigravity config was not copied. Reading
`~/Library/Application Support/Antigravity/User/History/.../9aD6.json` was blocked by the
credential-exploration guard, which is correct behavior: MCP config files hold API tokens. The
public endpoint needs no token for read-only Hub queries. If you want authenticated access (private
repos, higher rate limits), add your own token:

```bash
claude mcp remove huggingface --scope user
claude mcp add --transport http huggingface https://huggingface.co/mcp --scope user --header "Authorization: Bearer hf_YOUR_TOKEN"
```

## 2. Which model to use

### Candidates, with live Hub figures

| Model | Parameters | Released | Hub likes | Fits 6 GB for LoRA |
|---|---|---|---|---|
| `answerdotai/ModernBERT-base` | 149.7M | Jan 2025 | 1,099 | Yes, full fine-tune |
| `google/gemma-3-270m` | 268.1M | Aug 2025 | 1,098 | Yes, full fine-tune |
| `answerdotai/ModernBERT-large` | 395.9M | Jan 2025 | 483 | Yes, LoRA |
| `Qwen/Qwen2.5-0.5B` | 494.0M | Sep 2024 | 450 | Yes, LoRA |
| `Qwen/Qwen3-0.6B-Base` | 596.0M | Jul 2025 | 193 | Yes, LoRA |
| `Qwen/Qwen3.5-0.8B-Base` | 873.4M | Apr 2026 | 106 | Yes, LoRA, tight |

### Recommendation

Use `answerdotai/ModernBERT-base` as the primary and `Qwen/Qwen3.5-0.8B-Base` as the challenger.

ModernBERT-base is the right primary for four reasons. The task is closed-set classification over 20
fixed schemas with at most 5 options, which is what an encoder does well. It is bidirectional, so
option order bias and the causal-attention problems both drafts spend pages on largely disappear.
At 149.7M it is 3.3x smaller than Qwen2.5-0.5B, which is the only realistic path to the latency
number you already advertise. And Laya proves the family works on this exact benchmark at 0.766
accuracy, using ModernBERT-large.

Qwen3.5-0.8B-Base replaces the Qwen2.5-0.5B the Gemini draft names. Qwen2.5-0.5B shipped in
September 2024 and is two generations old. If you are going to pay for a decoder backbone, pay for
the current one.

Two notes against the obvious alternatives. ModernBERT-large (395.9M) is what Laya used, and going
straight to it means competing with Laya on its own ground with a smaller budget. Gemma-3-270m is
attractive on size but is a decoder trained for generation, so it inherits the same order-bias work
as Qwen with less community tooling for scoring heads.

Run both tracks in Phase 2 of the implementation plan and pick on measured accuracy at a latency
budget. The two tracks share everything except the backbone and the sequence builder, so the
comparison costs about a day.

### Hardware note for the 1660 Ti

The GTX 1660 Ti is Turing TU116, compute capability 7.5. Three consequences that will otherwise cost
you an afternoon:

* No bf16. Bfloat16 needs compute capability 8.0 or higher. Use fp16 with `torch.amp.GradScaler`,
  or fp32 for the heads. Any config that hardcodes `bf16=True` will fail.
* No FlashAttention-2. It also requires compute capability 8.0. Use PyTorch SDPA, which selects the
  memory-efficient backend on 7.5 and works fine at these sequence lengths.
* TU116 has no tensor cores, unlike the RTX 20-series. Packed fp16 math still runs at roughly twice
  the fp32 rate, so fp16 is worth using, but do not expect tensor-core speedups.

## 3. What Laya actually is

Pulled from the vendored source in `Laya/github-laya/laya/` and the live model cards.

Architecture: a bidirectional encoder (ModernBERT-large) plus a decision head trained from scratch,
421.3M parameters total. The sequence layout is
`[CLS] <type> instructions [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] state [SEP]`, capped at 512
tokens. A `[MASK]` token marks each option, and an MLP scorer reads the hidden state at each marker
and emits one scalar logit per option. That is a pointer head implemented through markers rather
than through query-key similarity.

Training objective (`laya/common.py::proper_reward`): a composite of log score, spherical score, and
ranked probability score for ordinal questions. It accepts soft target distributions, which matters
because this benchmark's labels are soft.

Calibration: per-bucket temperature scaling keyed on question type and cardinality
(`temp_bucket`). Its shipped fitted temperatures are 1.637, 1.251, and 1.983, all above 1.

There is also an `act_head`, a small MLP over `[top1, margin, entropy, k]` that predicts an action.
This is close to the correctness head proposed in `IMPLEMENTATION_PLAN.md` section 4.4, which is
useful evidence that the construction works in this family.

Inference cost: Laya re-encodes once per question. Its own latency table reads 38.4 ms for 1
question, 156.0 ms for 10, and 721.4 ms for 50, so cost is linear in question count. There is no
shared state prefix. That is a real weakness you can beat.

## 4. Are Laya and Kev benchmarked?

Laya: yes, and the real numbers are now confirmed from its published model card
(`convaiinnovations/laya-typed-decisions`, 421.3M, `model-index` present).

| Metric | Laya on typed-decisions |
|---|---|
| Accuracy | 0.766 |
| Soft accuracy | 0.509 |
| Brier | 0.066 |
| ECE | 0.214 |
| Score MAE | 0.242 |
| Within 1 level | 0.995 |
| Latency p50 | 158.3 ms per case |

Two corrections to the brief. The Brier is 0.066, not 0.119, and the ECE is 0.214, not 0.081. The
0.119 figure belongs to the ModernBERT-base row of Laya's comparison table.

This is the strongest possible confirmation of the feasibility analysis. Laya sits exactly on the
frontier predicted in `IMPLEMENTATION_PLAN.md` section 2: it matches the teacher panel well enough
to reach a Brier of 0.066, and it pays for that with an ECE of 0.214, which is 5.4x the target in
the brief and worse than Jev's 0.144. Nobody can have both. Laya chose Brier.

Kev: no. Kev has never been evaluated on typed-decisions, it is not listed on the community tracker
for Jev reproductions, and its published numbers come from its own suite (AG News, Banking77, BoolQ,
MNLI, SST-5, Yelp). Its measured Brier there is 0.302 and its macro accuracy is 0.78125. Treat Kev
as an architecture reference, which it is a good one, and not as a benchmark competitor.

### Where everything stands on the one harness

| Model | Accuracy | Brier | ECE | Latency p50 |
|---|---|---|---|---|
| Laya fine-tuned, 421M | 0.766 | 0.066 | 0.214 | 158.3 ms |
| TypeSafe Jev 1.13.0 (vendor claim) | 0.727 | 0.148 | 0.144 | 710 ms |
| Teacher self-agreement ceiling | 0.735 | n/a | n/a | n/a |
| ModernBERT-base specialist (claim) | 0.646 | 0.119 | 0.179 | 349 ms |
| TF-IDF plus logistic regression (measured here) | 0.6505 | 0.1445 | 0.0314 | under 1 ms |
| Verdict 1.0, 151M (measured here) | 0.2610 | 0.5851 | 0.4209 | 313.9 ms |
| Uniform random | 0.2690 | 0.2433 | 0.0485 | n/a |

The TF-IDF row holds the best ECE of anything on this table by a factor of 4.6, at 1/1000th the
latency, on thirty seconds of CPU. That is worth sitting with before committing a week.

## 5. Should you build Verdict 2.0, or pivot to Jev projects?

### What the tracker says about reach

`multimodalart/jev-reproductions-tracker` ranks 45 artifacts by
`X likes + 5 x GitHub stars + 8 x Hub likes + views / 500`. Categories: 17 decoding hacks,
15 trained models, 8 explainers, 3 diffusion, 2 prior art.

| Rank | Score | Kind | Artifact |
|---|---|---|---|
| 1 | 11,213 | explainer | 45-second TL;DR video on Jev |
| 2 | 10,541 | decoding | Qwen-2.5-1B parallel constrained decoding, built in 2 hours |
| 3 | 7,151 | decoding | openjev + openjev.com (1,427 GitHub stars) |
| 4 | 6,419 | trained | jevlike, a tiny option scorer |
| 9 | 1,679 | trained | Laya, 421M with full benchmarks |
| ~32 | 18 | trained | rlcd-modernbert-151m ("OpenJev Verdict"), yours |

Read those numbers carefully, because they do not say what you might hope.

The highest-reach artifact is a 45-second video. The second is a two-hour hack whose pitch was
"No new training required", and it took 1.1M views. Laya built a real 421M model with a real
benchmark and landed at 1,679, which is 6.3x below the two-hour hack. The most rigorous artifact on
the list, mini-jev, a preregistered measurement study, scored 65.

Reach in this niche tracks speed and narrative clarity. It does not track model quality. A Verdict
2.0 that goes from 0.766 to 0.815 accuracy is a marginal win that is hard to explain in one sentence,
in a lane where the incumbent already got modest reach.

### The one unclaimed slot

The tracker's own README names what is still missing from the open ecosystem: TypeSafe's weights,
the RLCD algorithm, and "any open model matching Jev's calibration claims."

That third item is open, and section 4 shows why nobody has taken it. Jev's ECE is 0.144. Laya's is
0.214. Every open reproduction chasing accuracy is getting worse at calibration, because the
benchmark's Brier metric pulls them toward the teacher distribution and away from calibrated
confidence. Nobody has published that this tradeoff exists, and you can prove it.

### Recommendation

Do all three, in this order, and treat the first as the product.

Step one, publish the analysis. You already own a result nobody else has: the benchmark everyone is
citing measures two incompatible quantities, and the frontier is computable. Laya's own model card
is the receipt (Brier 0.066 with ECE 0.214). This costs zero compute, it is defensible line by line,
and it is an explainer, which is the format that ranks first on the tracker. Lead with the claim
that the leaderboard is measuring two things at once, show the frontier table, and name the fix.

Step two, ship the model that implements the fix. A small ModernBERT-base with a separate confidence
channel, reporting both ECE figures honestly, targeting Brier at or below 0.105 and calibration-channel
ECE at or below 0.040. That is a specific, first-of-its-kind claim rather than another accuracy
increment. Building it also retires a liability: your tracker entry currently advertises "under 35ms"
for a model that measures 26.10% on the public benchmark, below the 26.90% you get from guessing.
Somebody will run that eval eventually.

Step three, use your Jev access as a measurement instrument, not as a project generator. Building
wrappers on the Jev API is the most crowded lane on the tracker, 17 of 45 artifacts, and the leader
has 1,427 GitHub stars. Being a wrapper makes you a distribution channel for someone else's product.
But you can do one thing nobody else on that list can: run the real Jev on the typed-decisions
harness and publish verified numbers. Every citation of "Jev 0.727 / Brier 0.148 / ECE 0.144" traces
back to a hardcoded Python literal in a Laya notebook. Replacing that literal with a measurement is
cheap, uniquely available to you, and makes you the reference everyone cites.

### One distribution fix, worth more than any of this

Your tracker entry has `handle: null`. There is no X account attached to it. Whatever reach that
Space generates currently flows nowhere. Open a PR against `index.html` in the Space to add your
handle before you publish anything else. That is a five-minute change against a page whose top entry
has 1.1M views.

## 6. Cost and hardware

### Cost

Zero dollars. Every step runs on hardware you own.

| Item | Cost |
|---|---|
| Base model weights (all candidates, Apache 2.0) | $0 |
| Dataset (already cached locally) | $0 |
| Training | $0 on your own GPU |
| Evaluation | $0 |
| Publishing to the Hub | $0 |

If you ever want a faster run, Kaggle gives 30 GPU-hours a week on 2x T4 at no cost, and that is
exactly what Laya used (`laya_finetune_typed_decisions_2xT4_kaggle.ipynb`). A paid A100 on RunPod or
Lambda runs about $1.10 to $1.50 an hour and the job is under an hour, so the worst case is roughly
$2. No account, no card, and no cloud GPU is required.

### Will it finish end to end on your systems?

Yes, on both, with the 1660 Ti as the faster option for training.

The job is small. With 840 training cases at about 516 packed tokens, 10 epochs is 8,400
forward-and-backward passes. At roughly 3 x 2 x tokens x active parameters, a Qwen-class 0.5B run is
about 9.3 PFLOP total, and a ModernBERT-base run is roughly a third of that.

| Machine | Track | Expected full training run |
|---|---|---|
| GTX 1660 Ti, 6 GB | ModernBERT-base, full fine-tune | 30 to 60 minutes |
| GTX 1660 Ti, 6 GB | Qwen3.5-0.8B-Base, LoRA r=16 | 2 to 4 hours |
| Apple Silicon, MPS | ModernBERT-base | 1 to 2 hours |
| Apple Silicon, MPS | Qwen3.5-0.8B-Base, LoRA | 3 to 6 hours |
| Kaggle 2x T4, free | either track | 20 to 45 minutes |

Memory on the 6 GB card, with fp16 weights and LoRA or a small full fine-tune:

* ModernBERT-base full fine-tune: about 2.1 GB including fp32 Adam states. Comfortable.
* Qwen2.5-0.5B with LoRA: about 1.0 GB of weights plus small adapter states. Comfortable.
* Qwen3.5-0.8B-Base with LoRA: about 1.75 GB of weights plus adapter states. Fits, with room for
  activations at batch 1 and gradient checkpointing on.

The 16 GB of system RAM is adequate. Data loading is trivial at this scale, since the entire
training set is 1,200 JSON records.

Two practical items before you start. Qwen weights are not in the local Hugging Face cache, so budget
a 1 to 2 GB download depending on which backbone you pick. And `peft` is not installed in `.venv`,
so add it if you take the LoRA path.

### What the schedule looks like

From `IMPLEMENTATION_PLAN.md` section 10, three to four working days total, of which under two hours
is GPU time on the primary path. Phase 0 (diagnosing why Verdict 1.0 scores below random) and Phase 1
(the eval harness and reference floors) are CPU-only and take about half a day combined. They are
also the two phases that produce the material for step one of the strategy above, so the analysis
you publish is a byproduct of work you need to do anyway.
