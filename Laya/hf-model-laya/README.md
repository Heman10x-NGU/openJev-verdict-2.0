---
license: apache-2.0
library_name: transformers
tags: [laya, rl-agent, system-one, calibrated-decisions, rlcd, classification, routing, scoring, reinforcement-learning, commercial-use]
---

# Laya

Laya is an open-source, non-autoregressive System 1 decision model: give it a **state** (text, email, ticket, or JSON document) and **typed questions**, and it returns typed answers with mathematically calibrated probabilities and confidence scores. It never generates text, eliminating parsing errors and hallucinations.

This is the **fine-tuned checkpoint**, incorporating dedicated email triage (spam, phishing, department routing), conversation trajectory modeling (TD(lambda = 1.0)), and per-cardinality temperature calibration.

| Question type | Returns |
|---|---|
| `choice` | Selected option, probabilities per option, calibrated confidence |
| `score` | Expected level on your ordinal rubric (0, 1, 2...), distribution, confidence |
| `noul` | Calibrated boolean probability P(true) from 0.0 to 1.0 |

## Architecture
- Backbone: `ModernBERT-large` (395M, fully fine-tuned, bidirectional), plus a decision head trained from scratch (2 transformer layers, an option marker scorer, and an act/escalate head). Total parameters: 421M.
- Option Markers: Every option is scored at its own `[MASK]` marker token, then a softmax is applied over that question's options.
- Input budget: 512 tokens per question (question + options + state).
- Multi-question batching: Evaluates all questions in a single forward pass (~33 to 38 ms on GPU).

## Training
Trained with **RLCD (Reinforcement Learning for Calibrated Decisions)**: the policy reports a probability distribution, exploration adds zero-mean Gaussian noise to the logits, and the reward is a strictly proper scoring rule (log score + spherical score, plus ranked probability score for ordinal score questions). The maximum expected reward is achieved only when the model outputs true, calibrated probabilities.

Multi-turn dialogues use Temporal Difference learning with Monte Carlo targets (TD(lambda = 1.0)) over prefix slices, preventing outcome leakage. 100% human-annotated real-world datasets, zero synthetic shortcuts.

- Training: Fine-tuned (7,313 updates, 1 epoch, ~1.96 hours)
- Fitted calibration temperatures: [1.637, 1.251, 1.983] (with per-option-count scaling)

## Benchmark: Laya vs. TypeSafe Jev

<div align="center">
  <img src="eval/benchmark_comparison.png" alt="Laya Benchmark Comparison" width="900" />
</div>

| Metric / Dimension | TypeSafe Jev (Published) | Laya (Fine-Tuned Checkpoint) | Analysis / Advantage |
|---|---|---|---|
| **P50 Latency (1 Question)** | ~400 ms avg (70 to 500 ms, 150 ms best) | **38.4 ms** (p95: 42.1 ms) | **Laya is ~10.4x faster on avg (4x faster than Jev best-case)** |
| **Batched Latency (10 Questions)** | ~1,500 ms (serial) / ~400 ms | **156.0 ms** (p95: 158.4 ms) | **Laya evaluates 10 questions in the time Jev answers 1** |
| **Batched Latency (50 Questions)** | Multi-second / rate-limited | **721.4 ms** | High-throughput parallel mini-batching |
| **Benchmark Accuracy** | **67.8%** (across 4 production workflows) | **83.8%** in-task macro accuracy | **Laya achieves +16.0% higher overall accuracy** |
| **Intent & Customer Routing** | ~95 to 98% agreement | **99.1% accuracy** (ECE: 0.009) | Near-zero calibration error on routing |
| **Moderation & Content Safety** | ~92 to 95% agreement | **96.7% accuracy** (ECE: 0.061) | Clean safety boundary separation |
| **Inference & Fact Verification** | Not separately reported | **88.3% accuracy** (ECE: 0.054) | Full bidirectional attention captures contradictions |
| **Instruction-Following Tasks** | Proprietary internal set | **87.8% in-task / 86.3% zero-shot** | Proven generalization across unseen tasks |
| **Email Triage & Phishing** | Vendor custom workflow | **73.2% accuracy** (ECE: 0.017) | Tailored email cleaning and phishing filters |
| **Selective Automation (@ 50% Cov)** | Claims human escalation | **92.2% accuracy** (ECE: 0.041) | Safe automated gating (confidence >= 0.85) |
| **Model Weights & Code** | Closed-source / proprietary API | **100% Open-source Apache 2.0** | Full data sovereignty and transparency |
| **Inference Cost** | $0.042 / 1M input tokens recurring | **$0.00 / self-hosted** | Runs on commodity GPUs, Mac MPS, or CPU |
| **Multi-Turn Trajectory Modeling** | Static state snapshots | **TD(lambda = 1.0) prefix modeling** | Real temporal credit assignment |
| **Deployment Mode** | Cloud-only egress | **Air-gapped / Local / On-Device** | Zero data egress (HIPAA/GDPR compliant) |

## Evaluation Results (This Checkpoint)
- **In-task test sets:** macro accuracy 0.838, macro ECE 0.060
- **Zero-shot (held-out task families):** macro accuracy 0.651, macro ECE 0.207
- **Key Task Families:**
  - Intent and routing: accuracy **0.991**, ECE 0.009
  - Moderation and safety: accuracy **0.967**, ECE 0.061
  - Emotion and tone: accuracy **0.906**, ECE 0.018
  - Email triage and phishing: accuracy **0.732**, ECE 0.017
  - Inference and fact checking: accuracy **0.883**, ECE 0.054

Detailed results by task family, reliability diagrams, and risk-coverage curves are located in `eval/` in this repository.

## Quickstart (`pip install laya`)

```python
import laya

# Load the model directly from Hugging Face Hub
agent = laya.load("convaiinnovations/laya")

state = {
    "from": "customer@acme.com",
    "subject": "Duplicate billing on March invoice #4411",
    "body": "Hi team, we were billed twice for March. Please refund the duplicate before Friday or we will cancel our plan."
}

questions = {
    "department": {
        "type": "choice",
        "instructions": "Which department should handle this email?",
        "criteria": {
            "billing": "invoices, payments, refunds",
            "technical": "bugs, outages, integrations",
            "sales": "pricing, contracts, demos",
            "other": "everything else"
        }
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this request?",
        "criteria": ["not urgent", "soon", "critical deadline or blocking issue"]
    },
    "churn_risk": {
        "type": "noul",
        "instructions": "Does the user threaten to cancel or switch to a competitor?"
    },
    "is_phishing": {
        "type": "noul",
        "instructions": "Is this email a phishing or scam attempt?"
    }
}

result = agent.predict(state, questions)
answers = result["answers"]

print("Department :", answers["department"]["choice"], f"(confidence: {answers['department']['confidence']:.2f})")
print("Urgency    :", f"{answers['urgency']['score']:.2f} / 2.0")
print("Churn Risk :", f"{answers['churn_risk']['noul']:.1%}")
print("Phishing   :", f"{answers['is_phishing']['noul']:.1%}")
```

## Interactive Web Demo
Try the live Gradio Space: [convaiinnovations/laya-demo](https://huggingface.co/spaces/convaiinnovations/laya-demo)

## License and Support
Released under the Apache 2.0 License by [Convai Innovations](https://huggingface.co/convaiinnovations), who offer commercial support, enterprise integration, and custom fine-tuning.

## Limitations
- Text only, English, 512 tokens per question (longer states are truncated).
- Calibration is measured on the benchmark datasets; evaluate on your own distribution before full automation.
- Arithmetic, counting, date comparisons, and multi-hop index lookups should be kept in deterministic code.
