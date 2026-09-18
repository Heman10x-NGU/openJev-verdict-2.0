# Verdict 2.0: Non-Autoregressive Decision Engine Specification

## Executive summary

Verdict 2.0 evaluates typed decision schemas over unstructured state context in a single, non-autoregressive forward execution. It discards conversational text generation in favor of bounded semantic judgment: categorical choices, ordinal scores, and binary probabilities. Every output includes calibrated confidence distributions, entropy metrics, and mathematically guaranteed conformal prediction sets, allowing application software to execute policy rules deterministically or escalate ambiguous cases to human review.

This specification unifies the functional strengths of TypeSafe Jev (typed functional primitives, explicit abstention, strict separation of semantic judgment from deterministic policy) and Laya (proper scoring rule optimization, dynamic option markers, and cardinality calibration). It eliminates the architectural bottlenecks of Verdict 1.0 (GLiClass bi-encoder similarity trap, ModernBERT sliding-window attention decoupling, and serial query re-encoding) while resolving all critical failure modes identified through adversarial red-teaming, systems auditing, and mathematical calibration theory:

* Two-Phase Decoupled Prefix-KV Engine for inference: Evaluates the state context once, storing a 48 MiB prefix KV-cache, and evaluates all $Q$ decision questions concurrently by expanding the cache across the batch dimension. Eliminates the 4.3 GiB activation memory explosion on Apple Silicon MPS, avoids $O(N^2)$ attention matrices, and runs natively on standard PyTorch SDPA, FlashAttention-2, TensorRT-LLM, and WebGPU WGSL kernels.
* Unified Gradient-Connected Graph for training: Preserves full end-to-end backpropagation through the state prefix and suffix branches during training using PyTorch 2.5+ FlexAttention block masks, switching to decoupled prefix-KV caching at inference time.
* FP32 Normalized Cosine Pointer Head with Cardinality Margin Scaling ($1 / \sqrt{\ln K}$): Prevents FP16 accumulator overflow ($> 65,504$) and cancels extreme-value log-sum-exp probability collapse across large option sets ($K \in [2, 64]$).
* Typed Loss Routing: Routes losses strictly by functional primitive type: Cross-Entropy with calibrated label smoothing for categorical choices, Cumulative Link Loss (Ordered Logistic Regression) for ordinal scores, and Binary Cross-Entropy for boolean propositions. Eliminates competing gradient vectors and boundary gradient vanishing.
* 4-Quadrant Counterfactual Twin-Pair Inoculation: Inoculates against Kev-0.5B's syntactic abstention shortcut by forcing the mutual information between the abstention string and label assignment to zero.
* Conformal Prediction Sets with Finite-Sample Coverage Guarantees: Provides split-conformal prediction sets $\mathcal{C}(x)$ guaranteeing $P(y^* \in \mathcal{C}(x)) \ge 1 - \alpha$ (such as 95% or 99%), giving software an exact mathematical threshold for autonomous dispatch versus escalation.
* Permutation-Invariant Option Evaluation: Eliminates multiple-choice order bias by isolating option tokens from sibling causal attention within the suffix, guaranteeing identical logits regardless of option display order.
* Production Logit Adjustment (Menon et al.): Supports post-hoc log-odds adjustment to account for base-rate prior shift between training distributions and production traffic.
* Pydantic v2 Native SDK: Built on PEP 593 `Annotated` types, providing static analysis under Mypy strict mode and Pyright, complete with dual-access `ExecutionResult[T]` containers.

```
+---------------------------------------------------------------------------------------------+
|                                    USER APPLICATION CODE                                    |
|  - Workflow Context: text string, log dump, or JSON document                                |
|  - Pydantic v2 Decision Schema: Choice[E], Score(min, max), BoolDecision                    |
+----------------------------------------------+----------------------------------------------+
                                               |
                                               v
+---------------------------------------------------------------------------------------------+
|                        PHASE 1: STATE PREFIX PREFILL (Length L_s)                           |
|  - Anchor BOS attention sink token at position 0                                            |
|  - Run state tokens once through Qwen2.5-0.5B backbone (24 layers, d=896, GQA 14:2)         |
|  - Cache prefix Key-Value tensors: K_prefix, V_prefix (Shape: [24, 1, 2, L_s, 64], ~48 MB)  |
+----------------------------------------------+----------------------------------------------+
                                               |
                                               v
+---------------------------------------------------------------------------------------------+
|                     PHASE 2: BATCHED SUFFIX FORWARD (Batch size Q)                          |
|  - Pack all Q decision questions into batch dimension [Q, L_q]                              |
|  - Expand K_prefix, V_prefix along batch dimension using zero-copy stride expansion         |
|  - Assign suffix RoPE position IDs starting strictly at L_s (shared origin across branches) |
|  - Standard causal cross-attention against prefix KV cache (100% native PyTorch SDPA)       |
+----------------------------------------------+----------------------------------------------+
                                               |
                                               v
+---------------------------------------------------------------------------------------------+
|                    CALIBRATED POINTER READOUT HEAD (FP32 Cosine Similarity)                 |
|  - Linear projection and LayerNorm in FP32: q = LN(W_Q h_field), k = LN(W_K h_option)       |
|  - Cardinality-scaled cosine similarity: z_k = (10.0 / sqrt(ln K)) * (q . k)                |
|  - Parametric continuous temperature scaling: T(K) = alpha * ln(K) + beta in [0.70, 2.50]   |
|  - Logit adjustment for production prior shift: z_tilde_k = z_k / T - tau * ln(pi_k)        |
+----------------------------------------------+----------------------------------------------+
                                               |
                                               v
+---------------------------------------------------------------------------------------------+
|                    CONFORMAL PREDICTION SET & POLICY DISPATCH ENGINE                        |
|  - result.values: Validated Pydantic model with native Enum, int, and bool types            |
|  - result.meta: Calibrated probabilities, prediction entropy, expected values, std dev      |
|  - result.conformal_set: Guaranteed prediction set C(x) with 95% coverage guarantee        |
|  - Deterministic policy engine: If |C(x)| == 1 auto-dispatch; else escalate to human review |
+---------------------------------------------------------------------------------------------+
```

---

## 1. Backbone and execution architecture

### 1.1 Model selection and parameters
Verdict 2.0 deploys two model tiers:
* Primary Flagship Tier: `Qwen/Qwen2.5-0.5B` (490M parameters). 24 transformer layers, hidden dimension $d = 896$, intermediate FFN dimension $d_{\text{ffn}} = 4,864$ (SwiGLU), 14 query heads, 2 key-value heads (Grouped Query Attention, 7:1 ratio), head dimension $d_h = 64$. Rotary Position Embeddings (RoPE) base frequency $\theta = 1,000,000$, supporting up to 32,768 tokens.
* Edge Mobile Tier: `answerdotai/ModernBERT-base` (149M parameters). 22 encoder layers, $d = 768$, 12 attention heads, unpadding flash-attention support, memory footprint under 350 MB in FP16.

### 1.2 Two-Phase Decoupled Prefix-KV Engine (Inference)
To evaluate $Q$ questions over a state context of length $L_s$, the engine runs in two phases rather than materializing a monolithic $N \times N$ attention tree:

1. Phase 1 (Prefix Prefill):
   The state text tokens $X_{\text{state}} \in \mathbb{R}^{1 \times L_s}$ (with an explicit BOS attention sink anchored at index 0) pass through the 24 transformer layers once. The resulting Key and Value representations are cached:
   $$\mathbf{K}_{\text{prefix}}, \mathbf{V}_{\text{prefix}} \in \mathbb{R}^{24 \times 1 \times 2 \times L_s \times 64}$$
   For a 4,096-token state context, this cache occupies exactly 48.00 MiB in FP16.

2. Phase 2 (Batched Suffix Forward):
   The $Q$ questions and candidate option tokens are packed into a batch tensor $X_{\text{suffixes}} \in \mathbb{R}^{Q \times L_q}$ (where $L_q = \max_k L_{q_k}$, right-padded to the longest question).
   The cached prefix KV tensors are expanded along the batch dimension using zero-copy stride manipulation:
   $$\mathbf{K}_{\text{prefix}}^{\text{expanded}} = \mathbf{K}_{\text{prefix}}.\text{expand}(24, Q, 2, L_s, 64)$$
   Each suffix attends to the shared state prefix and its own tokens causally. Sibling questions are physically isolated across batch rows, providing 100% mathematical isolation with zero attention mask overhead, zero $O(N^2)$ memory materialization, and full compatibility with native PyTorch SDPA, FlashAttention-2, and WebGPU WGSL kernels.

### 1.3 Unified Gradient-Connected Graph (Training)
During training, decoupling the forward pass into two independent calls with `past_key_values` detaches the backward computational graph, preventing task loss gradients from flowing into the state prefix encoder.
To maintain end-to-end differentiability:
* During training, sequences are packed into a single tensor using PyTorch 2.5+ `flex_attention` with `create_block_mask(tree_mask_mod)` and gradient checkpointing.
* The block mask computes FlashAttention tiled attention directly in SRAM, avoiding the 4.3 GiB intermediate activation tensor while propagating gradients through all 24 layers of the prefix encoder.

### 1.4 Strict RoPE position assignment
Suffix tokens must not use sequential position IDs based on flattened concatenation. In Phase 2, every question branch starts at position $L_s$:
$$\text{pos}(i) = \begin{cases} i & \text{for } 0 \le i < L_s \text{ (State tokens)} \\ L_s + j & \text{for token } j \text{ in question suffix } q \end{cases}$$
This ensures all $Q$ questions have identical relative rotary distance to the state context tokens, eliminating schema permutation instability.

### 1.5 Calibrated FP32 cosine pointer readout head
For dynamic candidate sets ($K \in [2, 64]$):
1. Representations $\mathbf{h}_{\text{field}} \in \mathbb{R}^{d}$ and $\mathbf{h}_{\text{opt}, k} \in \mathbb{R}^{d}$ are upcast to FP32.
2. Vectors are projected and normalized:
   $$\mathbf{q} = \text{LayerNorm}\left(\mathbf{W}_Q \mathbf{h}_{\text{field}}\right) \in \mathbb{R}^{d_{\text{proj}}}, \quad \mathbf{k}_k = \text{LayerNorm}\left(\mathbf{W}_K \mathbf{h}_{\text{opt}, k}\right) \in \mathbb{R}^{d_{\text{proj}}}$$
   where $d_{\text{proj}} = 256$.
3. Option logits $z_k$ use cardinality-normalized cosine similarity:
   $$z_k = \frac{\tau_0}{\sqrt{\ln K}} \cdot \frac{\mathbf{q}^\top \mathbf{k}_k}{\|\mathbf{q}\|_2 \|\mathbf{k}_k\|_2}$$
   where $\tau_0 = 10.0$. The factor $\frac{1}{\sqrt{\ln K}}$ cancels the extreme-value growth of distractor log-sum-exp accumulation.
4. Dummy padding tokens ($k > K$) are masked with $-1 \times 10^9$.
5. Logits are clamped to $[-50.0, 50.0]$ prior to softmax to prevent FP16 underflow and overflow:
   $$\mathbf{p} = \text{softmax}\left(\frac{\mathbf{z}}{T(K)}\right)$$

### 1.6 Permutation-invariant candidate option attention
In standard causal attention within suffixes, later options attend to earlier options, which introduces order bias (favoring earlier or later options).
In Verdict 2.0:
* Option tokens within the suffix attend to the question prompt and state prefix, but attention between candidate options is masked out.
* Each candidate option embedding is computed independently of the order in which options appear in the prompt. This mathematically enforces permutation equivariance in the pointer readout head.

---

## 2. Loss formulation and mathematical optimization

Verdict 2.0 routes loss computation by functional primitive type, avoiding gradient competition between cross-entropy and quadratic Brier scores.

### 2.1 Categorical choices (`Choice`)
For unordered discrete choices, the objective is Cross-Entropy with calibrated label smoothing ($\epsilon = 0.05$):
$$\mathcal{L}_{\text{Choice}} = -\sum_{k=1}^K y_k^* \ln p_k$$
Brier score is computed during validation to verify calibration, but is not mixed into the training loss as an additive scalar.

### 2.2 Ordinal scores (`Score`)
For ordered ordinal scales ($1 \dots K$), standard softmax and unweighted Ranked Probability Score (RPS) suffer from boundary gradient vanishing on extreme errors. Verdict 2.0 employs a Cumulative Link Model (Ordered Logistic Regression):
The model outputs a scalar score $s = \mathbf{w}^\top \mathbf{h}_{\text{field}} \in \mathbb{R}$ and learns monotonic thresholds $\theta_1 < \theta_2 < \dots < \theta_{K-1}$:
$$P(Y \le m) = \sigma(\theta_m - s) = \frac{1}{1 + e^{-(\theta_m - s)}}$$
$$P(Y = k) = P(Y \le k) - P(Y \le k-1)$$
The training loss is the exact negative log-likelihood:
$$\mathcal{L}_{\text{Score}} = -\ln P(Y = y^*)$$
Expected value and standard deviation are computed analytically from the resulting distribution:
$$\mathbb{E}[Y] = \sum_{k=1}^K k \cdot P(Y = k), \quad \sigma[Y] = \sqrt{\sum_{k=1}^K (k - \mathbb{E}[Y])^2 \cdot P(Y = k)}$$

### 2.3 Binary propositions (`BoolDecision`)
For binary decisions, the loss is standard binary cross-entropy:
$$\mathcal{L}_{\text{Bool}} = -y^* \ln p - (1 - y^*) \ln (1 - p)$$

### 2.4 Cardinality-aware continuous temperature calibration
Rather than fitting discrete step-function temperature buckets (which cause calibration cliffs at bucket borders), Verdict 2.0 fits a smooth parametric temperature function:
$$T(K) = \alpha \cdot \ln(K) + \beta, \quad T(K) \in [0.70, 2.50]$$
The parameters $\alpha$ and $\beta$ are fitted post-training on held-out validation data by minimizing Brier score:
$$\arg\min_{\alpha, \beta} \frac{1}{N} \sum_{n=1}^N \sum_{k=1}^{K_n} \left( \text{softmax}_k\left(\frac{\mathbf{z}_n}{T(K_n)}\right) - y_{k, n}^* \right)^2$$
The lower bound $T_{\min} \ge 0.70$ prevents the optimization from collapsing into Laya's pseudo-one-hot argmax exploit ($T = 0.1006$).

### 2.5 Production logit adjustment for prior shift
When production class distribution $\boldsymbol{\pi}_{\text{prod}}$ differs from training prevalence $\boldsymbol{\pi}_{\text{train}}$, raw logits are adjusted before softmax (Menon et al., 2021):
$$\tilde{z}_k = \frac{z_k}{T(K)} - \tau \cdot \ln\left(\frac{\pi_{\text{train}, k}}{\pi_{\text{prod}, k}}\right)$$
where $\tau \in [0.5, 1.0]$ controls the strength of prior re-balancing.

---

## 3. Conformal prediction sets with finite-sample guarantees

In mission-critical software, point estimates with soft probabilities force downstream code to guess when probabilities are split (such as $p_1 = 0.48, p_2 = 0.45$).
Verdict 2.0 implements Split Conformal Prediction (Angelopoulos & Bates, 2021) to produce a prediction set $\mathcal{C}(X) \subseteq \{1, \dots, K\}$ with a mathematically proven coverage guarantee:
$$P(Y \in \mathcal{C}(X)) \ge 1 - \alpha$$
where $\alpha \in (0, 1)$ is the user-specified error tolerance (such as $\alpha = 0.05$ for 95% coverage).

### 3.1 Non-conformity score and calibration
Using a held-out calibration set $(X_1, Y_1), \dots, (X_n, Y_n)$ of $n = 2,000$ examples:
1. Define the conformity score as the cumulative softmax probability of sorted classes up to the true class:
   $$s_i = \sum_{k=1}^{\text{rank}(Y_i)} p_{(k)}(X_i)$$
2. Compute the conformal threshold $\hat{q}$ as the $\lceil (n + 1)(1 - \alpha) \rceil / n$ empirical quantile of $\{s_1, \dots, s_n\}$.
3. At test time, the conformal prediction set $\mathcal{C}(X_{\text{test}})$ includes top classes until their cumulative probability clears $\hat{q}$:
   $$\mathcal{C}(X_{\text{test}}) = \{k \in \{1, \dots, K\} : \sum_{j=1}^{\text{rank}(k)} p_{(j)}(X_{\text{test}}) \le \hat{q} + p_{(k)}(X_{\text{test}})\}$$

### 3.2 Autonomous software policy dispatch rules
* Singleton set ($|\mathcal{C}(X)| = 1$): Unambiguous decision. Downstream software executes autonomous workflow actions with verified statistical confidence.
* Multi-element set ($|\mathcal{C}(X)| > 1$): Ambiguous decision. Software halts automated side effects and routes the specific candidate set to a human reviewer.
* Empty set ($|\mathcal{C}(X)| = \emptyset$): Out-of-distribution input. Software triggers an explicit abstention route (`__insufficient_evidence__`).

---

## 4. Training curriculum and data contamination protocol

### 4.1 Synthetic data volume and domain scaling
To prevent zero-shot generalization collapse, the synthetic curriculum is scaled to 50,000 diverse records:
* 500 distinct enterprise domains (ITSM, Observability, Cloud IAM, FinTech, Healthcare triage, Logistics, HR, Fraud detection, Security audits, etc.).
* 12,000 distinct schema topologies. No individual schema topology produces more than 10 training instances.
* Cardinality stratification: 20% binary ($K=2$), 30% small choice ($K \in [3, 5]$), 35% medium choice ($K \in [6, 12]$), 15% high cardinality ($K \in [13, 32]$).

### 4.2 Inoculation against Kev's syntactic shortcut bug
To eliminate the bug where models predict `"None of the above"` whenever that string appears in the candidate list, training data is generated using the 4-Quadrant Counterfactual Twin-Pair Algorithm:
* Twin A (Substantive Match): Context has an unambiguous match $t^*$. Candidates include $t^*$, $(K-2)$ hard distractors, and an abstention candidate $a \in \mathcal{A}$. Label $y_A^* = \text{index}(t^*)$.
* Twin B (True Abstention): Context has no match. Candidates include $(K-1)$ hard distractors and $a \in \mathcal{A}$. Label $y_B^* = \text{index}(a)$.
* Substantive Trap: A valid candidate contains negative lexical tokens (e.g., `"No prior history of service disruption"`). Label $y^* = \text{index}(c_{\text{substantive}})$.
Because the abstention string $a$ appears in 100% of Twin A and Twin B pairs, its lexical presence provides zero mutual information regarding whether it is the correct choice:
$$I\left(Y = k_{\text{abstain}}; \mathbf{1}[k_{\text{abstain}} \in \mathcal{C}]\right) = 0$$

### 4.3 Episodic abstention prevalence sampling
During Stage 2 fine-tuning, the batch-level abstention probability is sampled dynamically:
$$\alpha_{\text{batch}} \sim \text{Uniform}(0.05, 0.50)$$
This prevents the pointer head from learning a fixed prior intercept and forces reliance on contextual semantic evidence.

### 4.4 5-Tier cryptographic and semantic decontamination gate
Before any record enters training, it is verified against the held-out `LocalLLaMA/typed-decisions` test vault (400 cases, 2,000 questions) using five automated criteria:
1. Exact SHA-256 match on canonicalized Unicode text.
2. Character 5-gram and word 3-gram MinHash Jaccard similarity $\ge 0.50$.
3. Dense semantic embedding cosine similarity $\ge 0.84$ using `BGE-large-en-v1.5`.
4. Schema topology Jaccard overlap $\ge 0.40$ on candidate label sets.
5. Physical vault isolation: Test sets are stored in encrypted archives inaccessible to data generation scripts.

---

## 5. Developer SDK specification (Pydantic v2)

### 5.1 Schema definition ergonomics
Verdict 2.0 uses PEP 593 `Annotated` types to guarantee compatibility with Mypy strict mode, Pyright, and Pydantic v2 core schemas:

```python
from enum import Enum
from typing import Annotated, Any
from pydantic import BaseModel, Field

class ChoiceMeta:
    def __init__(self, allow_abstention: bool = True):
        self.allow_abstention = allow_abstention

class ScoreMeta:
    def __init__(self, min_score: int, max_score: int, step: int = 1):
        if min_score >= max_score:
            raise ValueError(f"min_score ({min_score}) must be less than max_score ({max_score})")
        self.min_score = min_score
        self.max_score = max_score
        self.step = step

class BoolMeta:
    pass

Choice = Annotated[Enum, ChoiceMeta()]
BoolDecision = Annotated[bool, BoolMeta()]

def Score(min_value: int, max_value: int, step: int = 1) -> Any:
    return Annotated[int, ScoreMeta(min_score=min_value, max_score=max_value, step=step)]
```

### 5.2 Application schema declaration
Policy escalation fields are decoupled from semantic judgment:

```python
class IncidentCategory(str, Enum):
    DATABASE_OUTAGE = "Database Outage: Connection pool exhaustion or disk failure"
    NETWORK_LATENCY = "Network Latency: Packet loss or routing degradation"
    APPLICATION_BUG = "Application Bug: Unhandled exception in application logic"
    AUTH_FAILURE = "Auth Failure: Credential rejection or token expiration"
    INSUFFICIENT_EVIDENCE = "__insufficient_evidence__: Context lacks necessary diagnostic logs"

class IncidentTriageSchema(BaseModel):
    category: Choice[IncidentCategory] = Field(
        description="Identify the root cause category"
    )
    severity: Score(1, 5) = Field(
        description="Assess business impact from 1 (cosmetic) to 5 (critical outage)"
    )
```

### 5.3 Output inspection and conformal policy dispatch

```python
# Execute decision pass with 95% conformal coverage guarantee
result = engine.decide(
    context=log_trace_text, 
    schema=IncidentTriageSchema,
    conformal_coverage=0.95
)

# 1. Direct typed value access
selected_category = result.values.category  # IncidentCategory.DATABASE_OUTAGE
selected_severity = result.values.severity  # 5

# 2. Conformal prediction set inspection
cat_set = result.conformal_sets["category"]
print(f"95% Conformal Set: {[opt.name for opt in cat_set.options]}")
print(f"Is unambiguous singleton: {cat_set.is_singleton}")

# 3. Deep calibration inspection
cat_meta = result.meta.choices["category"]
print(f"Top choice confidence: {cat_meta.confidence:.3f}, entropy: {cat_meta.entropy:.3f}")

sev_meta = result.meta.scores["severity"]
print(f"Expected severity: {sev_meta.expected_value:.2f} +/- {sev_meta.standard_deviation:.2f}")

# 4. Deterministic code controls policy (Invariant 4)
if cat_set.is_empty or cat_meta.is_abstained:
    route_to_human_triage(incident_id, reason="Explicit model abstention")
elif not cat_set.is_singleton:
    route_to_human_triage(
        incident_id, 
        reason=f"Ambiguous incident: candidates {[o.name for o in cat_set.options]}"
    )
elif sev_meta.expected_value >= 4.0:
    dispatch_emergency_pager(incident_id, target="oncall_dba")
else:
    create_standard_ticket(incident_id, queue=selected_category.name.lower())
```

### 5.4 Backward compatibility layer (`verdict.compat.v1`)
Verdict 2.0 includes a compatibility adapter for legacy Verdict 1.0, Kev, and Jev JSON request payloads (`POST /v1/systemone`):
* Translates dynamic dictionary criteria into runtime choice branches.
* Maps 0-indexed float scores to bounded integer ordinal distributions.
* Preserves legacy response formats without requiring microservice rewrites.

---

## 6. Verification protocol and performance targets

Verdict 2.0 is validated against the 400 held-out test cases (2,000 decisions) of `LocalLLaMA/typed-decisions` across all four enterprise domains:
* Top-1 exact accuracy: $\ge 81.5\%$ (surpassing Laya fine-tuned at 76.60% and Jev 1.13.0 at 72.70%).
* Soft accuracy (within tolerance): $\ge 65.0\%$.
* Brier score: $\le 0.1050$.
* Expected calibration error (ECE): $\le 0.0400$ under adaptive equal-mass binning.
* Conformal coverage empirical rate: $\ge 95.0\%$ under 95% target coverage.
* Inference latency (per 5-question case):
  * NVIDIA T4 (TensorRT INT8): $\le 20.0$ ms.
  * Apple Silicon M-Max: $\le 35.0$ ms.
  * Apple Silicon Base (with pre-cached state prefix): $\le 32.0$ ms.
  * Apple Silicon Base (cold full-pass prefill): $\le 160.0$ ms.
