# Executive summary: decision-native intelligence

This document outlines the product vision, economic thesis, and core architecture of **RLCD-demo**.

![Why Jev matters](assets/why-jev-matters.jpg)

## The core thesis

Most AI automation today relies on conversational Large Language Models (LLMs) generating strings. A software application submits a prompt, waits several seconds for an autoregressive decoder to emit tokens, parses the resulting text or JSON, handles format errors, and branches on the outcome.

This arrangement imposes high latency, high inference cost, and unpredictable parsing failures. In contrast, software typically requires a bounded judgment:
* Is this customer request an urgent churn risk?
* Which department owns this support ticket?
* Does this transaction satisfy our fraud threshold?

TypeSafe AI introduced this shift in September 2026 with **Jev**, a "System One" decision model that discards open-ended text generation entirely. Instead of generating sentences, Jev accepts unstructured input context and evaluates multiple typed questions in a single parallel forward pass, returning discrete choices, ordinal scores, and binary probabilities.

![How Jev works](assets/how-jev-works.png)

## Jevons' paradox: why efficiency drives volume

The model takes its name from William Stanley Jevons, the 19th-century British economist who observed that James Watt's more efficient steam engine led to an explosion in total coal consumption. By drastically lowering the cost of mechanical work, steam power expanded into previously unaffordable domains.

TypeSafe AI applies this logic to software automation:
* Current LLM calls cost $0.01 to $0.10 and take 1 to 10 seconds. Developers place them sparingly, usually once at the perimeter of an application.
* A decision-native model evaluates typed judgments in 70 to 300 milliseconds for $0.0004 per case.
* When semantic judgment becomes 100 times cheaper and faster, developers embed intelligence into every internal loop, webhook, automated test, and database trigger.

## The four-tier division of labor

Reliable automation splits responsibilities across specialized layers:

```
┌──────────────────────────────────────────────────────────┐
│ Tier 1: Deterministic Code                               │
│ State assembly, arithmetic, policy logic, side effects   │
└────────────────────────────┬─────────────────────────────┘
                             │
┌────────────────────────────▼─────────────────────────────┐
│ Tier 2: System 1 Decision Model (RLCD-demo)             │
│ Fast semantic categorization, typed choices, probabilities│
└────────────────────────────┬─────────────────────────────┘
                             │
┌────────────────────────────▼─────────────────────────────┐
│ Tier 3: Generative LLM                                   │
│ Drafting prose, synthesizing plans, user explanations    │
└────────────────────────────┬─────────────────────────────┘
                             │
┌────────────────────────────▼─────────────────────────────┐
│ Tier 4: Human Reviewer                                   │
│ Ambiguous, novel, high-risk, or escalated cases          │
└──────────────────────────────────────────────────────────┘
```

Deterministic software remains in control of state, policy rules, and external mutations. The decision model acts as a learned branch operator, evaluating messy inputs and returning calibrated uncertainty. Generative models run only when human-readable prose or complex creative synthesis is required.

## What TypeSafe AI disclosed

TypeSafe AI launched in September 2026, founded by Diogo Almeida (equal-contribution primary author of OpenAI's 2022 InstructGPT paper), Erik Gafni, and Sasha Sheng. Key verified disclosures include:
* Funding: Approximately $40 million seed funding led by DCVC.
* Valuation: Reported $200 million valuation.
* Published pricing: $0.042 per million input tokens, with unmetered output.
* Latency: 70 to 500 milliseconds end to end.
* Output primitives: `Choice` (up to 255 options), `Score` (ordered rubric), and `Noul` (binary yes/no proposition).
* Output formatting errors: 0% by construction, because the model projects predictions onto pre-declared software types rather than emitting text tokens.

## What remains unproven

TypeSafe AI has not published:
* A reproducible training recipe or algorithm paper for RLCD.
* Model weights, architecture specifications, or parameter counts.
* Empirical calibration curves (such as reliability diagrams or Brier scores) across public benchmarks.
* Performance under distribution shift or out-of-distribution inputs.

RLCD-demo exists to provide an open, transparent, and reproducible implementation of this architecture.
