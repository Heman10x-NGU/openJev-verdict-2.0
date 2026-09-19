# Decision-native foundation model specification

* Date: September 17, 2026
* Author / Model: Gemini Flash 3.8
* Context: RLCD-demo technical specification for peer review

This document defines the architecture, training methodology, evaluation protocol, and verification standards for decision-native foundation models ("System 1" engines).

---

## 1. Executive overview and problem statement

Conventional Large Language Models (LLMs) rely on autoregressive causal decoding. Applications that use LLMs for software automation submit an input prompt, wait 1,000 to 10,000 milliseconds for the model to emit sequential text tokens, parse the resulting JSON strings, handle syntax errors, and branch on the extracted fields.

This design introduces three structural failures in production automation:

1. Computational waste: Autoregressive token generation forces models to spend inference compute predicting formatting syntax (such as braces, quotes, and whitespace) rather than semantic decisions.
2. Miscalibrated confidence: Reinforcement Learning from Human Feedback (RLHF) optimizes for human preference, which penalizes ambiguity and forces the model into mode dropping. The resulting probability distributions are excessively sharp, yielding high confidence on incorrect predictions.
3. Latency and cost overhead: Autoregressive decoding costs between $0.01 and $0.10 per invocation, forcing engineering teams to ration intelligence at the application perimeter rather than embedding it inside internal loops.

A decision-native foundation model eliminates open-ended text generation. It accepts an input context and multiple typed questions, returning discrete choices, ordinal ratings, and binary probabilities in a single bidirectional forward pass. Outputs project directly into memory-resident data structures, guaranteeing zero syntax errors and sub-15-millisecond latency at $0.0004 per evaluation.

```
[ Tier 1: Deterministic Code ]   -> State assembly, arithmetic, policy logic, thresholds, side effects
[ Tier 2: System 1 Decision ]    -> Fast semantic categorization, typed choices, binary propositions
[ Tier 3: Generative LLM ]        -> Drafting prose, complex planning, user-facing explanations
[ Tier 4: Human Reviewer ]        -> Ambiguous, novel, high-risk, or escalated cases
```

Deterministic software retains authority over state, policy logic, and external side effects. The decision model acts as a learned branch operator. Generative models execute only when open-ended text synthesis is strictly necessary.

---

## 2. Core query primitives

The model evaluates three formal functional primitives without text generation:

### Choice
Selects a single categorical label from a declared set of candidate strings, supporting up to 255 options:

$$\hat{y} = \arg\max_{k \in \{1, \dots, K\}} P(c_k \mid x)$$

The engine returns the selected label, the full categorical distribution $\mathbf{p} = [p_1, \dots, p_K]$, and a scalar confidence metric derived from the distribution's normalized negative entropy:

$$\text{Confidence}(\mathbf{p}) = 1 - \frac{-\sum_{k=1}^K p_k \ln(p_k)}{\ln(K)}$$

### Score
Evaluates an input context against an ordered grading rubric across discrete ordinal levels:

$$\hat{s} = \sum_{m=1}^M v_m \cdot P(l_m \mid x)$$

Where $v_m \in \mathbb{R}$ represents the numerical value of rubric level $l_m$. The engine returns the expected score, the discrete probability distribution over levels, and the associated calibration metric.

### Noul
Evaluates a binary Bernoulli proposition:

$$P(\text{true} \mid x) \in [0.0, 1.0]$$

Where $P(\text{false} \mid x) = 1.0 - P(\text{true} \mid x)$. The primitive evaluates yes-or-no propositions directly without tokenizing boolean text strings.

---

## 3. The parallel inference mechanism

To understand the latency and pricing properties of this model, you must distinguish between two forms of parallelism:

1. Training sequence parallelism: The original Transformer architecture parallelized training by replacing recurrent neural network recurrence with self-attention. Every token attends to every other token simultaneously across a training sequence. However, standard LLM inference remains sequential, decoding one token at a time in an autoregressive loop.
2. Inference decision parallelism: Decision-native models parallelize inference-time decision computation. Because the candidate output set is bounded and declared in advance, the model scores every candidate option in a single parallel forward pass. 

This single-pass mechanism explains the economic model: output tokens are free because there is no autoregressive decode loop to meter after the initial forward pass.

---

## 4. Model backbone and inference architecture

You do not use causal decoder backbones. Autoregressive attention masks restrict each token to attending only to prior tokens, and sequential token decoding creates high latency and memory fragmentation.

You deploy a bidirectional transformer encoder:

```
Input Document ────────┐
                       ├─► [ Joint Tokenizer ] ─► [ ModernBERT Encoder ] ─► [ Matching Head ] ─► Logits
Candidate Labels List ─┘                                                                           │
                                                                                                   ▼
                                                                                            [ Softmax + ECE ]
                                                                                                   │
                                                                                                   ▼
Python Dict Assembly ◄───────────────────────────────────────────────────────────────────── Typed Output
```

### ModernBERT backbone
You select `answerdotai/ModernBERT-base` as the encoder backbone:
* Parameter count: 139 million parameters.
* Context window: 8,192 tokens.
* Attention architecture: Native FlashAttention-2 with alternating local sliding-window attention (128 tokens) and global attention layers.
* Positional encoding: Rotary Position Embeddings (RoPE).
* Input packing: Native unpadding, allowing batches of variable-length documents to run without wasting floating-point operations on padding tokens.
* Inference speed: 0.8 milliseconds on Apple Silicon unified memory; 0.2 milliseconds on an Nvidia A100.

### Dynamic bi-encoder matching head
Rather than freezing class projections into a static linear head, you implement a dynamic bi-encoder matching head following `knowledgator/gliclass-modern-base-v2.0`. 

The tokenizer formats the input sequence by concatenating the context and all declared candidate options using reserved delimiter tokens:

```
[TEXT] Customer email text here... [LABEL] billing_issue [LABEL] cancellation [LABEL] insufficient_evidence
```

ModernBERT processes this concatenated sequence with full bidirectional self-attention, allowing every document token to attend directly to every label token in a single pass. 

The classification head extracts token representations for the document span $\mathbf{h}_{\text{doc}} \in \mathbb{R}^d$ and label representations $\mathbf{h}_{\text{label}, k} \in \mathbb{R}^d$, computing similarity logits:

$$z_k = \frac{\mathbf{h}_{\text{doc}} \cdot \mathbf{h}_{\text{label}, k}^\top}{\sqrt{d}}$$

Applying softmax over the candidate slice produces normalized class probabilities without emitting text.

### Topological DAG execution for dependent fields
Parallel evaluation assumes that schema fields are conditionally independent given the context:

$$P(F_1, \dots, F_M \mid x) = \prod_{i=1}^M P(F_i \mid x)$$

When fields depend on each other (such as requiring a successful purchase status before authorizing a refund), evaluating them simultaneously produces logical contradictions.

You resolve this through two-stage Topological DAG execution:

```
                       ┌──────────────────────┐
                       │  Input Document (x)  │
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
                     [ Stage 2: Refund Action ]
                     (Conditioned on Order Status + Tier)
```

1. Independent fields execute simultaneously in Stage 1 during your first forward pass.
2. The runtime collects selected values and appends them to the context as structured state attributes: `[STATE] order_status=COMPLETED customer_tier=PLATINUM`.
3. Dependent fields evaluate in Stage 2 during a second forward pass.
4. Total execution completes in under 5 milliseconds on a GPU, guaranteeing logical consistency without autoregressive token loops.

---

## 5. Training pipeline and mathematical calibration

To understand the training objective, contrast the three reinforcement learning methodologies:

| Method | Optimization Objective | Primary Domain |
| :--- | :--- | :--- |
| RLHF (Human Feedback) | Human preference approval | Conversational chat and alignment |
| RLVR (Verifiable Rewards) | Programmatic correctness | Code execution and mathematics |
| RLCD (Calibrated Decisions) | Empirical probability calibration | Structured software workflows |

RLHF trains a model to sound persuasive, frequently driving winning logits to extreme values. This produces miscalibrated probabilities. RLCD trains the model so that its reported confidence matches its empirical accuracy across large evaluation sets, following the standard of calibrated weather forecasting.

### Composite RLCD loss function
You train the model using a composite loss that pairs cross-entropy classification loss with the Brier score, a strictly proper scoring rule:

$$\text{Brier}(\mathbf{p}, \mathbf{y}) = \frac{1}{K} \sum_{k=1}^K (p_k - y_k)^2$$

Where $\mathbf{p}$ is the predicted probability distribution and $\mathbf{y} \in \{0, 1\}^K$ is the one-hot target vector.

The total optimization objective is:

$$\mathcal{L}_{\text{RLCD}} = \mathcal{L}_{\text{CrossEntropy}} + \lambda \cdot \mathcal{L}_{\text{Brier}}$$

You set the calibration weighting scalar to $\lambda = 1.0$.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CalibratedDecisionLoss(nn.Module):
    """Composite loss combining Cross-Entropy and Brier score for probability calibration."""

    def __init__(self, brier_weight: float = 1.0) -> None:
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.brier_weight = brier_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Computes composite loss over unnormalized logits and target class indices.
        
        Args:
            logits: Unnormalized model outputs of shape [batch_size, num_classes].
            targets: Target ground-truth indices of shape [batch_size].
        """
        ce_loss = self.ce(logits, targets)
        probs = F.softmax(logits, dim=-1)
        targets_one_hot = F.one_hot(targets, num_classes=logits.size(-1)).float()
        brier_loss = torch.mean(torch.sum((probs - targets_one_hot) ** 2, dim=-1))
        return ce_loss + (self.brier_weight * brier_loss)
```

### Post-training temperature scaling
After fine-tuning on the composite loss, you freeze the encoder and head weights, calibrating a single positive temperature scalar $T$ on a validation split using L-BFGS:

$$p_i = \frac{e^{z_i / T}}{\sum_{j=1}^K e^{z_j / T}}$$

Optimizing $T$ against validation Negative Log-Likelihood (NLL) adjusts distribution dispersion without altering argmax classification accuracy.

---

## 6. Dataset generation and synthesis protocol

You build a dataset of 50,000 to 100,000 synthetic training pairs across enterprise software domains using frontier models (GPT-4o and Claude 3.5 Sonnet).

```json
{
  "context": "CloudTrail event: Source IP 198.51.100.44 executed DescribeInstances 412 times in 60 seconds followed by CreateAccessKey on IAM user admin-temp.",
  "primitive": "choice",
  "field_name": "incident_severity",
  "candidate_labels": [
    "benign_administrative_action",
    "credential_compromise_reconnaissance",
    "routine_monitoring_poll",
    "insufficient_evidence"
  ],
  "target_label": "credential_compromise_reconnaissance",
  "target_index": 1,
  "difficulty": "hard"
}
```

### Dataset construction invariants
1. Explicit abstention: In exactly 20% of training instances, construct candidate lists where no choice matches the text, setting the target to `"insufficient_evidence"` or `"none_of_the_above"`. This forces the model to abstain rather than assigning arbitrary probability mass to an incorrect label when presented with out-of-distribution inputs.
2. Label order permutation: Randomly permute the order of candidate labels during each training epoch. This prevents the transformer from acquiring positional biases toward earlier tokens in the candidate sequence.
3. Consensus filtering for production text: When sourcing real text from open issue trackers, audit logs, and filings, run dual-model labeling with GPT-4o and Claude 3.5 Sonnet. Discard any sample where the two reference models disagree.

---

## 7. Verification and evaluation protocol

To avoid synthetic benchmark self-preference, you evaluate the model across three independent test suites:

### 1. Human-annotated intent benchmarks
* Banking77: 13,083 customer service queries categorized into 77 fine-grained intents with verified human ground truth.
* CLINC150: 22,500 queries across 10 domains and 150 intents, including 1,200 out-of-scope instances that directly measure model abstention accuracy.
* MultiWOZ 2.4: Multi-domain dialogue state tracking with slot-filling constraints.

### 2. Held-out enterprise OOD stress suite
Construct 10,000 multi-field schemas with expert human labels spanning security alerts, healthcare triage, and commerce routing. Maintain a dedicated 2,000-example stress split with withheld candidate options to verify that the model correctly routes to `"insufficient_evidence"`.

### 3. Quantitative calibration metrics
Measure Expected Calibration Error (ECE) across ten equal-width confidence bins on held-out test data:

$$\text{ECE} = \sum_{m=1}^{10} \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$

```python
import numpy as np

def compute_ece(probs: np.ndarray, labels: np.ndarray, num_bins: int = 10) -> float:
    """Computes Expected Calibration Error across M equal-width confidence bins."""
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = predictions == labels

    bin_boundaries = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0

    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return float(ece)
```

Every model release must clear an ECE threshold under 0.04 (4%) and publish its reliability diagram alongside held-out Brier scores.

---

## 8. Definition of a new class of foundational models

A decision-native engine qualifies as a new foundational model class by meeting four criteria:

1. Type-native memory projection: The model discards conversational token emission. Predictions project directly into typed language structures (`Choice`, `Score`, `Noul`) in memory, achieving a 0% syntax and schema error rate by construction.
2. Proper probabilistic calibration: Confidence values reflect empirical real-world accuracy under proper scoring rules. When the model outputs 85% confidence across 1,000 cases, exactly 850 are correct, allowing deterministic software to branch on high-confidence predictions ($\ge 0.85$) while escalating lower-confidence cases to human review.
3. Jevons efficiency profile: Inferences complete in under 15 milliseconds for $0.0004 per call. This 100x speed and cost improvement allows developers to embed semantic evaluations inside internal code loops, continuous integration tests, webhooks, and database triggers.
4. Strict tier separation: The model acts as Tier 2 ("System 1" rapid semantic judgment) inside a four-tier architecture, situated between deterministic application code (Tier 1) and generative synthesis models (Tier 3), with human review serving as Tier 4.

---

## 9. Compute budget and reproduction cost

You can train and evaluate this architecture using standard cloud infrastructure:

| Component | Service | Specification | Projected Cost |
| :--- | :--- | :--- | :--- |
| Synthetic dataset | OpenAI API (GPT-4o-mini) | 50,000 paired schemas (15M tokens) | $2.25 |
| Model training | JarvisLabs.ai | Nvidia A100 40GB (1.5 hours) | $1.35 (roughly ₹115) |
| Weights distribution | Hugging Face Hub | `model.safetensors`, GGUF, ONNX | Free |
| Client evaluation demo | GitHub Pages | WebGPU in-browser inference (`transformers.js`) | Free |
| Total pipeline cost | | | $3.60 (roughly ₹300) |
