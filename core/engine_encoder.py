"""Primary non-autoregressive decision engine using ModernBERT and GLiClass.

Evaluates typed decision queries (Choice, Score, Noul) in a single batched
forward pass over dynamic labels with explicit abstention candidates and
properly calibrated confidence scores.
"""

from __future__ import annotations

import json
import logging
import math
import os
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
        max_length: int = 512,
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

        self.ort_session = None
        if model != "mock" and model is None and self.device == "cpu":
            try:
                import onnxruntime as ort
                onnx_file = None
                if isinstance(model_name_or_path, str) and os.path.isdir(model_name_or_path):
                    cand = os.path.join(model_name_or_path, "model.onnx")
                    if os.path.exists(cand):
                        onnx_file = cand
                if not onnx_file:
                    repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
                    for cand in [
                        os.path.join(repo_dir, "artifacts", "v2", "model.onnx"),
                        os.path.join(repo_dir, "artifacts", "openjev_modernbert.onnx"),
                        os.path.join(repo_dir, "..", "artifacts", "v2", "model.onnx"),
                        os.path.join(repo_dir, "..", "artifacts", "openjev_modernbert.onnx"),
                        "artifacts/v2/model.onnx",
                        "artifacts/openjev_modernbert.onnx",
                    ]:
                        if os.path.exists(cand):
                            onnx_file = cand
                            break
                if onnx_file and os.path.exists(onnx_file):
                    sess_opts = ort.SessionOptions()
                    sess_opts.intra_op_num_threads = 4
                    sess_opts.inter_op_num_threads = 1
                    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                    self.ort_session = ort.InferenceSession(
                        onnx_file,
                        sess_opts,
                        providers=["CPUExecutionProvider"],
                    )
                    logging.info("Preferred FP32 ONNX session loaded from %s", onnx_file)
            except Exception as exc:
                logging.warning("Failed to initialize ONNX session; falling back to PyTorch: %s", exc)
                self.ort_session = None

        if self.calibrator is None:
            cal_file = None
            if isinstance(model_name_or_path, str) and os.path.isdir(model_name_or_path):
                for fname in ["calibrator.json", "calibrator_modernbert.json"]:
                    cand = os.path.join(model_name_or_path, fname)
                    if os.path.exists(cand):
                        cal_file = cand
                        break
            if not cal_file:
                repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
                for cand in [
                    os.path.join(repo_dir, "artifacts", "v2", "calibrator.json"),
                    os.path.join(repo_dir, "artifacts", "calibrator.json"),
                    os.path.join(repo_dir, "artifacts", "calibrator_modernbert.json"),
                    os.path.join(repo_dir, "..", "artifacts", "v2", "calibrator.json"),
                    os.path.join(repo_dir, "..", "artifacts", "calibrator.json"),
                    os.path.join(repo_dir, "..", "artifacts", "calibrator_modernbert.json"),
                    "artifacts/v2/calibrator.json",
                    "artifacts/calibrator.json",
                    "artifacts/calibrator_modernbert.json",
                ]:
                    if os.path.exists(cand):
                        cal_file = cand
                        break
            if not cal_file and isinstance(model_name_or_path, str) and not os.path.exists(model_name_or_path):
                try:
                    from huggingface_hub import hf_hub_download
                    cal_file = hf_hub_download(model_name_or_path, "calibrator.json")
                except Exception:
                    pass
            if cal_file and os.path.exists(cal_file):
                try:
                    self.calibrator = TemperatureCalibrator.load(cal_file)
                    try:
                        with open(cal_file, "r", encoding="utf-8") as f:
                            cal_data = json.load(f)
                        if "per_k" in cal_data:
                            self.calibrator.per_k = cal_data["per_k"]
                    except Exception:
                        pass
                    logging.info("Auto-loaded calibrator from %s (T=%s)", cal_file, self.calibrator.temperature)
                except Exception as exc:
                    logging.warning("Failed to auto-load calibrator from %s: %s", cal_file, exc)

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
                logging.warning(
                    "Candidate count %d exceeds MAX_SUPPORTED_CANDIDATES %d; degrading by truncating.",
                    len(labels),
                    MAX_SUPPORTED_CANDIDATES,
                )
                labels = labels[:MAX_SUPPORTED_CANDIDATES]
                ids = ids[:MAX_SUPPORTED_CANDIDATES]
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

            raw_logits = None
            if self.ort_session is not None:
                try:
                    ort_inputs = {
                        "input_ids": tokenized_inputs["input_ids"].cpu().numpy(),
                        "attention_mask": tokenized_inputs["attention_mask"].cpu().numpy(),
                    }
                    ort_outs = self.ort_session.run(None, ort_inputs)
                    raw_logits = torch.from_numpy(ort_outs[0]).to(self.device)
                except Exception as exc:
                    logging.warning("ONNX execution failed (%s); falling back to PyTorch", exc)
                    raw_logits = None

            if raw_logits is None:
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
                per_k = getattr(self.calibrator, "per_k", None)
                k_val = str(len(ids))
                if per_k and k_val in per_k:
                    t_val = float(per_k[k_val])
                    cal_logits = logits_tensor / t_val
                else:
                    try:
                        cal_logits = self.calibrator(logits_tensor, k=len(ids))
                    except TypeError:
                        cal_logits = self.calibrator(logits_tensor)
                cal_status = "calibrated_for_scope"
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
