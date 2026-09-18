"""Secondary causal LLM adapter for non-autoregressive decision readout.

Evaluates decision primitives in a single forward pass by reading the logits
at the final position for verified single-token answer slots. Strictly enforces
prefix-preserving token boundaries and forbids continuation loops.
"""

from __future__ import annotations

import inspect
import time
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from core.calibration import TemperatureCalibrator
from core.primitives import (
    Choice,
    ChoiceResult,
    DecisionBatchResult,
    DecisionResult,
    INSUFFICIENT_EVIDENCE_DESC,
    INSUFFICIENT_EVIDENCE_ID,
    Noul,
    NoulResult,
    Option,
    Query,
    Score,
    ScoreResult,
)

LETTERS: tuple[str, ...] = (
    "A", "B", "C", "D", "E", "F", "G", "H",
    "I", "J", "K", "L", "M", "N", "O", "P",
)


class TokenBoundaryError(ValueError):
    """Raised when answer-slot tokens fail single-token roundtrip or prefix boundary checks."""


class CausalDecisionAdapter:
    """Evaluates decision primitives using last-position logits of a causal LLM."""

    def __init__(
        self,
        model: Any = None,
        tokenizer: Any = None,
        calibrator: TemperatureCalibrator | None = None,
        device: str = "cpu",
        model_id: str = "causal-adapter",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.calibrator = calibrator
        self.device = device
        self.model_id = model_id

    @staticmethod
    def get_slot_tokens(tokenizer: Any, count: int) -> list[int]:
        """Validate that slots A..P encode to exactly one token with round-trip equality."""
        if count > len(LETTERS):
            raise ValueError(
                f"Requested {count} slots, but maximum supported slots is {len(LETTERS)}."
            )
        slot_tokens: list[int] = []
        for letter in LETTERS[:count]:
            encoded = tokenizer.encode(letter, add_special_tokens=False)
            if len(encoded) != 1:
                raise TokenBoundaryError(
                    f"Slot letter {letter!r} encoded to {len(encoded)} tokens, expected exactly 1."
                )
            decoded = tokenizer.decode(encoded)
            if decoded != letter:
                raise TokenBoundaryError(
                    f"Slot letter {letter!r} failed exact roundtrip decode: got {decoded!r}."
                )
            slot_tokens.append(encoded[0])

        if len(slot_tokens) != len(set(slot_tokens)):
            raise TokenBoundaryError("Answer-slot tokens collide with each other.")

        return slot_tokens

    def render_prompt_and_validate_boundaries(
        self,
        context: str,
        query: Query,
    ) -> tuple[list[int], list[int], list[str], bool]:
        """Construct full prompt with chat template and verify token boundaries.

        Returns:
            prompt_token_ids: Token IDs of the prompt up to generation point.
            slot_token_ids: Token IDs for each slot letter.
            candidate_option_ids: Option/Outcome IDs corresponding to each slot.
            is_noul: Whether this query is a Noul proposition.
        """
        # Build candidate options with explicit abstention
        options_list: list[tuple[str, str]] = []  # (id, description)
        if query.kind == "choice":
            for opt in query.options:
                options_list.append((opt.id, opt.description))
            options_list.append((INSUFFICIENT_EVIDENCE_ID, INSUFFICIENT_EVIDENCE_DESC))
            question_text = query.question
        elif query.kind == "score":
            for lvl in query.levels:
                options_list.append((lvl.id, f"{lvl.description} (Value: {lvl.value})"))
            options_list.append((INSUFFICIENT_EVIDENCE_ID, INSUFFICIENT_EVIDENCE_DESC))
            question_text = query.question
        elif query.kind == "noul":
            options_list = [
                ("true", "True: Supported by sufficient evidence."),
                ("false", "False: Contradicted by sufficient evidence."),
                (INSUFFICIENT_EVIDENCE_ID, INSUFFICIENT_EVIDENCE_DESC),
            ]
            question_text = f"Evaluate proposition: {query.proposition}"
        else:
            raise TypeError(f"Unsupported query kind: {query}")

        total_candidates = len(options_list)
        slot_token_ids = self.get_slot_tokens(self.tokenizer, total_candidates)

        # Render options block
        rendered_options: list[str] = []
        candidate_option_ids: list[str] = []
        for idx, (opt_id, opt_desc) in enumerate(options_list):
            letter = LETTERS[idx]
            rendered_options.append(f"{letter}. {opt_desc}")
            candidate_option_ids.append(opt_id)

        options_str = "\n".join(rendered_options)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an autonomous decision engine. "
                    "Analyze the context and select the single letter corresponding to the correct option. "
                    "Output only the single choice letter."
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion:\n{question_text}\n\nOptions:\n{options_str}",
            },
        ]

        # Render prompt
        if hasattr(self.tokenizer, "apply_chat_template"):
            prompt_str = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            # Fallback simple template
            prompt_str = f"Context:\n{context}\n\nQuestion:\n{question_text}\n\nOptions:\n{options_str}\n\nAnswer:"

        prompt_token_ids = self.tokenizer.encode(prompt_str, add_special_tokens=False)

        # Assert prefix equality for EVERY candidate slot:
        # encode(prompt + slot) == encode(prompt) + [slot_token]
        for letter, slot_token in zip(LETTERS[:total_candidates], slot_token_ids):
            extended_ids = self.tokenizer.encode(
                prompt_str + letter, add_special_tokens=False
            )
            expected_ids = prompt_token_ids + [slot_token]
            if extended_ids != expected_ids:
                raise TokenBoundaryError(
                    f"Prompt boundary corrupted tokenization for slot {letter!r}: "
                    f"expected suffix [{slot_token}], got {extended_ids[len(prompt_token_ids):]}"
                )

        return prompt_token_ids, slot_token_ids, candidate_option_ids, query.kind == "noul"

    def _forward_last_logits(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Execute single forward pass and extract logits at the final position."""
        kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "use_cache": False,
            "return_dict": True,
        }
        forward_params = inspect.signature(self.model.forward).parameters
        if "logits_to_keep" in forward_params:
            kwargs["logits_to_keep"] = 1

        outputs = self.model(**kwargs)
        logits = outputs.logits
        # Extract last position: shape (batch_size, vocab_size)
        return logits[:, -1, :].float()

    def evaluate(self, context: str, queries: Sequence[Query]) -> DecisionBatchResult:
        """Evaluate a batch of decision queries in a single forward pass."""
        if not queries:
            return DecisionBatchResult(
                results=(),
                forward_call_count=0,
                total_latency_ms=0.0,
                execution_mode="single_call",
            )

        start_time = time.perf_counter()
        batch_prompt_ids: list[list[int]] = []
        batch_slot_ids: list[list[int]] = []
        batch_option_ids: list[list[str]] = []

        for q in queries:
            p_ids, s_ids, opt_ids, _ = self.render_prompt_and_validate_boundaries(
                context=context,
                query=q,
            )
            batch_prompt_ids.append(p_ids)
            batch_slot_ids.append(s_ids)
            batch_option_ids.append(opt_ids)

        # Pad sequences to max length (left padding is standard for causal models)
        max_len = max(len(ids) for ids in batch_prompt_ids)
        pad_token_id = getattr(self.tokenizer, "pad_token_id", 0) or 0

        padded_input_ids = []
        padded_masks = []
        for ids in batch_prompt_ids:
            pad_len = max_len - len(ids)
            padded_ids = [pad_token_id] * pad_len + ids
            mask = [0] * pad_len + [1] * len(ids)
            padded_input_ids.append(padded_ids)
            padded_masks.append(mask)

        inputs_tensor = torch.tensor(padded_input_ids, dtype=torch.long, device=self.device)
        masks_tensor = torch.tensor(padded_masks, dtype=torch.long, device=self.device)

        with torch.inference_mode():
            vocab_logits = self._forward_last_logits(inputs_tensor, masks_tensor)

        results: list[DecisionResult] = []
        total_latency_ms = (time.perf_counter() - start_time) * 1000.0

        for idx, q in enumerate(queries):
            slots = batch_slot_ids[idx]
            opt_ids = batch_option_ids[idx]

            # Slice only the logits corresponding to declared candidate slots
            candidate_logits = vocab_logits[idx, slots].unsqueeze(0)  # (1, num_candidates)

            # Apply calibrator if present
            if self.calibrator is not None:
                cal_logits = self.calibrator(candidate_logits)
                cal_status = "calibrated_conditional"
            else:
                cal_logits = candidate_logits
                cal_status = "uncalibrated_conditional"

            raw_probs = F.softmax(cal_logits, dim=-1)[0].cpu().tolist()
            p_sum = sum(raw_probs)
            probs = [float(p / p_sum) for p in raw_probs]
            prob_dict = {opt_id: float(p) for opt_id, p in zip(opt_ids, probs)}

            max_idx = int(torch.argmax(cal_logits[0]).item())
            selected_id = opt_ids[max_idx]
            selected_prob = prob_dict[selected_id]
            is_abstention = (selected_id == INSUFFICIENT_EVIDENCE_ID)

            if q.kind == "choice":
                results.append(
                    ChoiceResult(
                        id=q.id,
                        selected_id=selected_id,
                        selected_probability=selected_prob,
                        probabilities=prob_dict,
                        is_abstention=is_abstention,
                        model_id=self.model_id,
                        calibration_status=cal_status,
                        latency_ms=total_latency_ms / len(queries),
                    )
                )
            elif q.kind == "score":
                substantive_mass = sum(
                    prob_dict[lvl.id] for lvl in q.levels if lvl.id in prob_dict
                )
                if substantive_mass > 0.0 and not is_abstention:
                    # Conditional expected score over substantive levels
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
                        model_id=self.model_id,
                        calibration_status=cal_status,
                        latency_ms=total_latency_ms / len(queries),
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
                        selected_outcome=selected_id,  # "true", "false", or "__insufficient_evidence__"
                        p_true_given_sufficient_evidence=p_true_cond,
                        p_insufficient_evidence=p_abstain,
                        probabilities=prob_dict,
                        is_abstention=is_abstention,
                        model_id=self.model_id,
                        calibration_status=cal_status,
                        latency_ms=total_latency_ms / len(queries),
                    )
                )

        return DecisionBatchResult(
            results=tuple(results),
            forward_call_count=1,
            total_latency_ms=total_latency_ms,
            execution_mode="single_call",
        )
