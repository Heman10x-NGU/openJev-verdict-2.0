# Generalization experiment suite

## Why this exists

A 6-example probe on authored phishing emails returned:

| Checkpoint | Correct | Failure mode |
| :--- | ---: | :--- |
| Base `gliclass-modern-base-v2.0`, zero-shot | 5 / 6 | one over-abstention |
| Banking77 fine-tune (shipped v2) | 1 / 6 | abstained on 5 of 6 at confidence 0.58 to 0.80 |

The fine-tuned model did not confuse phishing with legitimate mail. It refused to
answer. Training on Banking77 with 13% abstention-gold examples appears to have
taught a domain gate ("not a banking intent, therefore abstain") rather than a
transferable decision skill. That is a mechanistic explanation for experiment E9,
where the same checkpoint scored 48.07% on TypeSafe's public evaluation cases.

n = 6 and the emails were authored, so this is a hypothesis, not a finding. This
suite tests it properly on public data.

The unifying design across every experiment below: **run both checkpoints on
every task and report both columns.** The question is not "can this model do
phishing". It is "what did domain fine-tuning cost in generality, measured". That
is a result other people building the confidence-cascade pattern need, and nobody
has published it.

---

## Cookbook triage

Twenty-four cookbooks in `RLCD Cookbook/`. Ranked by whether they can be
replicated locally with public data and a 25-logit Choice head, with no TypeSafe
API key.

### Tier 1: build these

| Experiment | Cookbook | Data (verified reachable) | K | Why |
| :--- | :--- | :--- | ---: | :--- |
| **G1** Phishing and fraud email | `patterns_confidence_routing.md` | `ealvaradob/phishing-dataset`, `SetFit/enron_spam` | 3 | Matches the viral cascade demo. Binary plus abstention is K=3, the model's strongest cardinality (97% on banking). Directly comparable to the timeline. |
| **G2** Jailbreak and hazard screening | `llm_guardrails.md` | `TrustAIRLab/in-the-wild-jailbreak-prompts` for positives, `tatsu-lab/alpaca` for benign | 3 | Same K=3 shape. High-demand use case. Cookbook uses Noul, which `core/formatting.py` already renders as a 3-way Choice. |
| **G3** Hierarchical beam-search classification | `hierarchical_classification.md` | Shopify `product-taxonomy` categories.txt | <= 25 per node | **Highest value.** Turns the 25-candidate ceiling into a feature: a 10,000-leaf taxonomy becomes a sequence of small Choices. Directly answers the repo's most-cited published limitation. |
| **G4** Function and tool routing | `function_calling.md` | Authored fixture, 10 to 20 typed functions | 10-20 | Matches the agent use cases in the thread (pre-screening, tool selection). Cheap to build, needs an owned fixture with a committed manifest. |

### Tier 2: only if Tier 1 lands early

| Experiment | Cookbook | Blocker |
| :--- | :--- | :--- |
| G5 Passage re-ranking | `rerank_typesafe.md` | `jhu-clsp/CLERC` is reachable but 30 candidates needs chunking into two passes. |
| G6 SEC industry classification | `classification_using_confidence.md` | 75 groups needs the G3 hierarchy machinery. EDGAR acquisition is real work. Strong conceptual fit with the confidence curve. |
| G7 Skill selection | `skill_suggestion.md` | Hermes roster (182 skills) is public, but the full loop needs an Anthropic key. The ranking half is standalone and testable. |

### Not worth building

`citation_check`, `date_extraction`, `entity_alignment`, `autoformat`,
`semantic_find`, `classifying_rag_passages`, `autoresearch_feature_discovery`,
`consistency_*`, `parallel_questions`. Each needs either bespoke authored data or
the `Score` and `Noul` primitives, which the repository itself documents as
experimental and unbenchmarked. Do not showcase an unbenchmarked primitive.

---

## Experiment specifications

Every experiment writes `reports/v2/gen_<id>_<name>.json` and every one reports
both checkpoints.

### Shared protocol

1. Sample size at least 500 per task, balanced unless the task has a natural
   prevalence, with the seed committed.
2. Report for each checkpoint: accuracy, NLL, Brier, ECE equal-width and
   adaptive, abstention rate, abstention precision and recall, and a 1000-sample
   bootstrap 95% CI on accuracy and abstention rate.
3. Report the **selective-risk curve** for each, thresholds 0.50 to 0.99, because
   that is the artifact this project exists to produce.
4. Report the **over-abstention rate**: share of items where the model abstained
   although the correct option was present in the candidate list. This is the
   metric that the probe suggests is the whole story.
5. Never tune a prompt, a threshold, or a label description to improve a number
   after seeing it. If label wording is varied, that is its own recorded arm with
   all variants reported.

### G1: phishing and fraud email

Two candidate descriptions plus abstention. Draw 500 from
`ealvaradob/phishing-dataset` and 500 legitimate from `SetFit/enron_spam`. Commit
the exact row ids.

Additional arms, because the probe suggests label wording matters more than the
task:
- `banking_framed`: describe the options in banking-support language.
- `neutral_framed`: describe them in generic security language.

Report both. If the fine-tune recovers under banking framing, that is direct
evidence for the domain-gate hypothesis and is the most interesting result in the
suite.

Then run the cascade: at each threshold, escalation rate, retained accuracy, and
errors above the gate, for both checkpoints. This is the table that answers the
unanswered replies on the viral cascade post.

### G2: jailbreak and hazard screening

500 jailbreak prompts from `TrustAIRLab/in-the-wild-jailbreak-prompts`, 500
benign instructions from `tatsu-lab/alpaca`. K=3.

Report the full ROC, not only accuracy, because the operating point matters more
than the headline for a guardrail. Report the false-block rate on benign traffic
at the threshold that catches 95% of jailbreaks.

### G3: hierarchical beam-search classification

Fetch the Shopify taxonomy. Build a tree where every node has at most 24 children
plus abstention, splitting wider nodes into alphabetical sub-groups.

Implement `core/hierarchy.py` with greedy and beam search (K configurable),
carrying the product of path probabilities. Compare against the flat baseline
wherever the level fits in 25 logits.

Report: leaf accuracy, accuracy at each depth, forward passes per document,
wall-clock per document, and where in the tree errors concentrate. Compare greedy
against beam K=3 and K=5.

This converts "the head caps at 25 choices" from a limitation into a documented
scaling strategy. It is the single highest-leverage item in the suite.

### G4: function and tool routing

Author 20 typed functions with argument enums, and 300 natural-language requests
covering them plus 60 requests that no function serves (abstention gold). Commit
the fixture with a provenance manifest marking it project-authored, following the
convention `oss/open-jev-v2/openjev` uses.

Report accuracy, abstention recall on the uncovered requests, and the cascade
curve.

---

## What to do with the outcome

If the hypothesis holds and the fine-tune under-performs the base checkpoint out
of domain, the response is not to hide it. It is to ship both:

- `verdict-banking-151m`: the specialist, 95.0% in domain.
- `verdict-base-151m`: the base checkpoint with the calibration harness and
  documented zero-shot behaviour.

And publish the tradeoff as the finding: domain fine-tuning a zero-shot decision
model bought X points in domain and cost Y points everywhere else, measured
across four task families. That is a genuinely useful result for anyone deploying
the cascade pattern, and it is a stronger contribution than any single accuracy
number in the repository.

If the hypothesis fails and the fine-tune generalises fine, then the probe was
noise from six authored examples and the suite becomes four new domains where the
model works. That is also a good outcome.

Either way the experiment is worth running, which is the point.
