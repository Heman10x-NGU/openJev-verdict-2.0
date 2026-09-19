# Architecture and system specification

This document specifies the architecture, functional primitives, model selection rationale, and schema execution pipeline for **RLCD-demo**.

## The functional primitives

RLCD-demo implements the three core query primitives defined by TypeSafe AI:

### 1. Choice
Selects one label from a declared set of candidate strings (supporting up to 255 options).
* Input: Shared context string, field description, and list of candidate strings.
* Output: Selected label, probability distribution across all candidates, and a scalar confidence value.
* Example: Selecting ticket department from `["billing", "security", "infra", "sales", "unsupported"]`.

### 2. Score
Evaluates the context against an ordered grading rubric or descriptive levels.
* Input: Shared context string, rubric description, and ordered levels.
* Output: Predicted continuous score (or ordinal level), probability distribution across levels, and a confidence value.
* Example: Evaluating customer frustration level from `1` (calm) to `5` (furious).

### 3. Noul
Evaluates a binary proposition (named after the Bernoulli distribution).
* Input: Shared context string and a proposition statement.
* Output: Probability $P(\text{true}) \in [0.0, 1.0]$.
* Example: Evaluating whether an email requests an immediate refund.

---

## Model backbone: bidirectional encoder versus autoregressive decoder

Do not use causal decoder models (such as Qwen, Llama, or Mistral) for non-text decision tasks. Causal masking prevents tokens from attending to subsequent tokens, and the autoregressive decode loop introduces latency and memory fragmentation.

RLCD-demo uses a modern bidirectional transformer encoder:

### Primary backbone: ModernBERT (`answerdotai/ModernBERT-base`)
* Parameter count: 139 million parameters.
* Attention mechanism: Native FlashAttention-2 with alternating local and global attention.
* Context window: 8,192 tokens.
* Padding efficiency: Native unpadding (processing variable-length sequences without wasting compute on padding tokens).
* Compute cost: 0.8 milliseconds per forward pass on Apple Silicon unified memory; 0.2 milliseconds on an Nvidia A100.

### Pre-trained zero-shot reference: `knowledgator/gliclass-modern-base-v2.0`
* Parameter count: Roughly 200 million parameters.
* Architecture: ModernBERT-base combined with a GLiNER-style bi-encoder classification head.
* Capability: Encodes the text context and all candidate labels into a joint representation, producing classification logits for all candidates in a **single forward pass**.

---

## Non-autoregressive joint encoding

Instead of generating JSON syntax strings, the model projects semantic representations directly into candidate label logits.

```
Input Document ────────┐
                       ├─► [ Joint Tokenizer ] ─► [ ModernBERT Encoder ] ─► [ Bi-Encoder Head ] ─► Logits
Candidate Labels List ─┘                                                                            │
                                                                                                    ▼
                                                                                             [ Softmax + ECE ]
                                                                                                    │
                                                                                                    ▼
Python Dict Assembly ◄────────────────────────────────────────────────────────────────────── Typed JSON
```

1. Document context and candidate labels are concatenated with specialized delimiter tokens (`[TEXT]`, `[LABEL]`).
2. ModernBERT processes all tokens with full bidirectional attention, allowing every document token to attend to every label token.
3. The classification head computes similarity scores between document span representations and label token embeddings.
4. Softmax normalizes probabilities across the candidate slice.
5. Python memory constructs the structured dictionary, eliminating JSON syntax errors.

---

## Solving the dependent-fields problem

A key technical failure in previous hacks was the assumption of field independence ($P(F_1, \dots, F_M \mid C) = \prod P(F_i \mid C)$). Predicting fields in parallel without conditioning produces contradictory outputs (for example, generating `refund: true` when `order_status: "CANCELLED"`).

RLCD-demo addresses this through **Topological DAG Execution**:

```
                       ┌──────────────────────┐
                       │  Input Document (C)  │
                       └──────────┬───────────┘
                                  │
                  ┌───────────────┴───────────────┐
                  ▼                               ▼
          [ Stage 1: Order Status ]       [ Stage 1: Customer Tier ]
          (Independent parallel pass)     (Independent parallel pass)
                  │                               │
                  └───────────────┬───────────────┘
                                  │
                                  ▼
                     [ Stage 2: Refund Route ]
                     (Conditioned on Order Status + Tier)
```

1. Independent fields execute simultaneously in Stage 1 using a single parallel forward pass.
2. Dependent fields declare prerequisite parent fields in their schema definition.
3. In Stage 2, parent selections append to the context as structured state attributes (`[STATE] order_status=COMPLETED`), and dependent fields evaluate in a second fast forward pass.
4. Total execution requires two fast passes (under 5 milliseconds total on GPU), preserving logical consistency while remaining 50 times faster than autoregressive LLM decoding.
