"""Primary non-autoregressive decision engine using ModernBERT and GLiClass.

Evaluates typed decision queries (Choice, Score, Noul) in a single batched
forward pass over dynamic labels with explicit abstention candidates and
properly calibrated confidence scores.
"""

from __future__ import annotations

import math
import time
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from core.calibration import TemperatureCalibrator
from core.formatting import (
    CapacityError,
    MAX_SUPPORTED_CANDIDATES,
    build_model_input,
    format_prompt,
    format_query,
)
from core.primitives import (
    ChoiceResult,
    DecisionBatchResult,
    DecisionResult,
    INSUFFICIENT_EVIDENCE_DESC,
    INSUFFICIENT_EVIDENCE_ID,
    NoulResult,
    Query,
    ScoreResult,
)


def _compute_concentration(probs: Sequence[float]) -> float:
    """Compute normalized negative entropy: 1.0 - (H(p) / ln(K)).

    Returns a distribution shape statistic in [0.0, 1.0] where 1.0 is a point mass
    and 0.0 is a uniform distribution.
    """
    k = len(probs)
    if k <= 1:
        return 1.0
    entropy = -sum(p * math.log(p) for p in probs if p > 1e-12)
    max_entropy = math.log(k)
    return max(0.0, min(1.0, 1.0 - (entropy / max_entropy)))


class DecisionEngine:
    """Non-autoregressive decision engine evaluating typed queries concurrently."""

    def __init__(
        self,
        model_name_or_path: str = "knowledgator/gliclass-modern-base-v2.0",
        model: Any = None,
        tokenizer: Any = None,
        calibrator: TemperatureCalibrator | None = None,
        device: str = "cpu",
        max_length: int = 1024,
    ):
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.max_length = max_length
        self.calibrator = calibrator

        if model is not None and tokenizer is not None:
            self.model = model
            self.tokenizer = tokenizer
        elif model == "mock":
            self.model = None
            self.tokenizer = None
        else:
            # Lazy or eager load from HuggingFace
            from gliclass import GLiClassModel
            from transformers import AutoTokenizer

            self.model = GLiClassModel.from_pretrained(model_name_or_path)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            self.model.to(self.device)
            self.model.eval()

    def evaluate(
        self,
        context: str,
        queries: Sequence[Query],
    ) -> DecisionBatchResult:
        """Evaluate multiple typed queries in a single batched forward pass."""
        if not queries:
            return DecisionBatchResult(
                results=(),
                forward_call_count=0,
                total_latency_ms=0.0,
                execution_mode="single_call",
            )

        start_time = time.perf_counter()

        formatted_texts: list[str] = []
        batch_labels: list[list[str]] = []
        batch_ids: list[list[str]] = []

        for q in queries:
            text, labels, ids = format_query(context, q)
            if len(labels) > MAX_SUPPORTED_CANDIDATES:
                raise CapacityError(len(labels), MAX_SUPPORTED_CANDIDATES)
            formatted_texts.append(text)
            batch_labels.append(labels)
            batch_ids.append(ids)

        if self.model is None:
            # Mock mode for testing without GPU/checkpoint dependencies
            batch_logits = []
            for labels in batch_labels:
                row_logits = [3.0] + [0.5] * (len(labels) - 1)
                batch_logits.append(row_logits)
        else:
            prompts = []
            for q, labels, text in zip(queries, batch_labels, formatted_texts):
                if q.kind in ("choice", "score"):
                    prompts.append(build_model_input(q.question, context, labels))
                else:
                    prompts.append(build_model_input("", text, labels))
            tokenized_inputs = self.tokenizer(
                prompts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)

            with torch.inference_mode():
                outputs = self.model(**tokenized_inputs)
                raw_logits = outputs.logits  # shape: (batch_size, max_num_classes)

            if not torch.all(torch.isfinite(raw_logits)):
                raise ValueError("Model output contains non-finite logits (NaN or Inf).")

            batch_logits = []
            for i, labels in enumerate(batch_labels):
                num_classes = len(labels)
                candidate_logits = raw_logits[i, :num_classes].float().cpu().tolist()
                batch_logits.append(candidate_logits)

        # Process each field prediction
        results: list[DecisionResult] = []
        total_latency_ms = (time.perf_counter() - start_time) * 1000.0
        per_query_latency = total_latency_ms / len(queries)

        for idx, q in enumerate(queries):
            logits_tensor = torch.tensor(
                [batch_logits[idx]], dtype=torch.float32, device=self.device
            )
            ids = batch_ids[idx]

            # Temperature calibration and scope determination
            if self.calibrator is not None:
                cal_logits = self.calibrator(logits_tensor)
                # Check whether current query matches evaluated calibration scope
                if getattr(self.calibrator, "scope", "") == "restricted_5_candidate_selection" and len(ids) == 5:
                    cal_status = "calibrated_for_scope"
                else:
                    cal_status = "unvalidated_scope"
            else:
                cal_logits = logits_tensor
                cal_status = "uncalibrated"

            raw_probs = F.softmax(cal_logits, dim=-1)[0].cpu().tolist()
            # Normalize to sum to exactly 1.0 within floating point precision
            p_sum = sum(raw_probs)
            probs = [float(p / p_sum) for p in raw_probs]
            prob_dict = {opt_id: p for opt_id, p in zip(ids, probs)}

            max_idx = int(torch.argmax(cal_logits[0]).item())
            selected_id = ids[max_idx]
            selected_prob = prob_dict[selected_id]
            is_abstention = (selected_id == INSUFFICIENT_EVIDENCE_ID)
            concentration = _compute_concentration(probs)

            if q.kind == "choice":
                results.append(
                    ChoiceResult(
                        id=q.id,
                        selected_id=selected_id,
                        selected_probability=selected_prob,
                        probabilities=prob_dict,
                        is_abstention=is_abstention,
                        concentration=concentration,
                        model_id=self.model_name_or_path,
                        calibration_status=cal_status,
                        latency_ms=per_query_latency,
                    )
                )
            elif q.kind == "score":
                substantive_mass = sum(
                    prob_dict[lvl.id] for lvl in q.levels if lvl.id in prob_dict
                )
                if substantive_mass > 0.0 and not is_abstention:
                    expected_val = sum(
                        lvl.value * (prob_dict[lvl.id] / substantive_mass)
                        for lvl in q.levels
                    )
                    selected_val = next(
                        (lvl.value for lvl in q.levels if lvl.id == selected_id), None
                    )
                else:
                    expected_val = None
                    selected_val = None

                results.append(
                    ScoreResult(
                        id=q.id,
                        selected_level_id=selected_id,
                        selected_value=selected_val,
                        expected_score=expected_val,
                        probabilities=prob_dict,
                        is_abstention=is_abstention,
                        abstention_probability=prob_dict.get(INSUFFICIENT_EVIDENCE_ID, 0.0),
                        model_id=self.model_name_or_path,
                        calibration_status=cal_status,
                        latency_ms=per_query_latency,
                    )
                )
            elif q.kind == "noul":
                p_abstain = prob_dict.get(INSUFFICIENT_EVIDENCE_ID, 0.0)
                substantive_mass = prob_dict.get("true", 0.0) + prob_dict.get("false", 0.0)
                if substantive_mass > 0.0 and not is_abstention:
                    p_true_cond = prob_dict.get("true", 0.0) / substantive_mass
                else:
                    p_true_cond = None

                results.append(
                    NoulResult(
                        id=q.id,
                        selected_outcome=selected_id,  # type: ignore[arg-type]
                        p_true_given_sufficient_evidence=p_true_cond,
                        p_insufficient_evidence=p_abstain,
                        probabilities=prob_dict,
                        is_abstention=is_abstention,
                        model_id=self.model_name_or_path,
                        calibration_status=cal_status,
                        latency_ms=per_query_latency,
                    )
                )

        return DecisionBatchResult(
            results=tuple(results),
            forward_call_count=1,
            total_latency_ms=total_latency_ms,
            execution_mode="single_call",
        )
