# RLCD training and calibration recipe

This document provides the mathematical foundation, training objectives, and PyTorch implementation for **Reinforcement Learning for Calibrated Decisions (RLCD)**.

## The calibration problem

Standard language models trained with cross-entropy loss optimize for discriminatory accuracy: they maximize the logit of the winning class relative to competitors. This objective encourages overconfidence. A model will frequently output 0.99 probability for a class it is only 60% certain about, because cross-entropy rewards large logit separations.

In autonomous software workflows, overconfidence is catastrophic:
* If software branches on confidence thresholds ($\ge 0.85$), an overconfident model triggers automated actions on flawed data.
* If a model claims 80% confidence across 1,000 decisions, it must be correct in exactly 800 instances.

RLCD solves this by combining discriminatory cross-entropy loss with a strictly proper scoring rule.

---

## Proper scoring rules and the Brier score

A scoring rule is strictly proper if its expected value is uniquely minimized when the model's reported probability distribution $P$ equals the true underlying data distribution $Q$.

The **Brier score** (introduced by Glenn Brier in 1950) is the mean squared error between predicted probabilities and one-hot outcome vectors:

$$\text{Brier}(p, y) = \frac{1}{K} \sum_{k=1}^K (p_k - y_k)^2$$

Where $K$ is the number of candidate classes, $p_k$ is the model's predicted probability for class $k$, and $y_k \in \{0, 1\}$ is the ground truth.

### The composite RLCD loss function

To maintain high classification accuracy while penalizing miscalibrated probability distributions, train the model with a composite loss:

$$\mathcal{L}_{\text{RLCD}} = \mathcal{L}_{\text{CrossEntropy}} + \lambda \cdot \mathcal{L}_{\text{Brier}}$$

Where $\lambda$ is a calibration weighting hyperparameter (recommended setting: $\lambda = 1.0$).

---

## PyTorch implementation: `CalibratedDecisionLoss`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CalibratedDecisionLoss(nn.Module):
    """Composite loss combining Cross-Entropy and Brier Score for probability calibration."""
    
    def __init__(self, brier_weight: float = 1.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.brier_weight = brier_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Unnormalized raw logits of shape [batch_size, num_classes]
            targets: Class indices of shape [batch_size]
        """
        # Cross-entropy loss for classification accuracy
        ce_loss = self.ce(logits, targets)
        
        # Softmax probabilities over candidates
        probs = F.softmax(logits, dim=-1)
        
        # One-hot encoded ground truth
        targets_one_hot = F.one_hot(targets, num_classes=logits.size(-1)).float()
        
        # Brier score across candidate classes
        brier_loss = torch.mean(torch.sum((probs - targets_one_hot) ** 2, dim=-1))
        
        return ce_loss + (self.brier_weight * brier_loss)
```

---

## Synthetic dataset generation recipe

To give the model zero-shot decision capabilities across arbitrary schemas, create 50,000 to 100,000 synthetic training pairs using an existing frontier LLM.

### Dataset schema
Every training instance must follow this JSON structure:

```json
{
  "context": "Customer email: The recurring charge of $49 appeared twice on my Visa card ending in 4012. Please reverse the second charge immediately.",
  "primitive": "choice",
  "field_name": "triage_category",
  "candidate_labels": [
    "billing_duplicate_charge",
    "cancellation_request",
    "account_compromise",
    "feature_inquiry",
    "insufficient_evidence"
  ],
  "target_label": "billing_duplicate_charge",
  "target_index": 0,
  "difficulty": "medium"
}
```

### Critical dataset requirements
1. Explicit abstention: In exactly 20% of instances, generate schemas where none of the candidates match the text, and set the ground-truth target to `"insufficient_evidence"` or `"none_of_the_above"`. This forces the model to abstain rather than assign artificial confidence to an irrelevant label.
2. Label order permutation: Randomly shuffle the order of candidate labels during each training epoch to prevent positional bias.
3. Domain variety: Generate cases across security logs, support tickets, medical notes, pull request diffs, and financial receipts.

---

## Post-training temperature scaling

After fine-tuning the model backbone on the composite loss, freeze all model weights and run **Temperature Scaling** on a validation set (established by Guo et al., 2017).

Temperature scaling rescales logits by a single positive scalar $T$:

$$p_i = \frac{e^{z_i / T}}{\sum_j e^{z_j / T}}$$

```python
import torch
import torch.nn as nn
from torch.optim import LBFGS

class ModelWithTemperature(nn.Module):
    """Post-processing calibration wrapper using a single learned temperature scalar."""
    
    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base_model = base_model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, input_ids, attention_mask):
        logits = self.base_model(input_ids, attention_mask=attention_mask)
        return self.temperature_scale(logits)

    def temperature_scale(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature

    def calibrate(self, val_loader, device: str = "cuda"):
        """Optimizes temperature T on validation data using Negative Log Likelihood."""
        self.to(device)
        nll_criterion = nn.CrossEntropyLoss().to(device)
        optimizer = LBFGS([self.temperature], lr=0.01, max_iter=50)

        # Collect validation logits and labels
        logits_list, labels_list = [], []
        with torch.no_grad():
            for batch in val_loader:
                logits = self.base_model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
                logits_list.append(logits)
                labels_list.append(batch["labels"].to(device))
        
        logits = torch.cat(logits_list)
        labels = torch.cat(labels_list)

        def eval_step():
            optimizer.zero_grad()
            loss = nll_criterion(self.temperature_scale(logits), labels)
            loss.backward()
            return loss

        optimizer.step(eval_step)
        print(f"Optimal calibration temperature: {self.temperature.item():.4f}")
```

---

## Measuring calibration: Expected Calibration Error (ECE)

To prove your model's reliability, calculate Expected Calibration Error on a held-out test split:

```python
import numpy as np

def compute_ece(probs: np.ndarray, labels: np.ndarray, num_bins: int = 10) -> float:
    """Computes Expected Calibration Error across M confidence bins."""
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = predictions == labels

    bin_boundaries = np.linspace(0, 1, num_bins + 1)
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

A target ECE below 0.04 (4%) demonstrates production-grade probabilistic calibration.
