"""Centralized prompt and label formatting contracts for RLCD and GLiClass.

Guarantees identical formatting across training, inference, ONNX export,
and browser WebGPU/WASM runtimes.
"""

from __future__ import annotations

from typing import Sequence

from core.primitives import (
    Choice,
    INSUFFICIENT_EVIDENCE_DESC,
    INSUFFICIENT_EVIDENCE_ID,
    Noul,
    Query,
    Score,
)

LABEL_MARKER = "<<LABEL>>"
SEP_MARKER = "<<SEP>>"
LABEL_TOKEN_ID = 50368
SEP_TOKEN_ID = 50369

MAX_SUPPORTED_CANDIDATES = 25
MAX_SUBSTANTIVE_CANDIDATES = 24


class CapacityError(ValueError):
    """Raised when a query exceeds the maximum supported candidate capacity."""

    def __init__(self, count: int, max_allowed: int = MAX_SUPPORTED_CANDIDATES):
        super().__init__(
            f"Candidate count {count} exceeds maximum model capacity of {max_allowed}."
        )
        self.count = count
        self.max_allowed = max_allowed


INPUT_TEMPLATE: str = "Question: {question}\n\nContext:\n{context}"


def format_prompt(text: str, label_descriptions: Sequence[str]) -> str:
    """Format prompt with label prefix and separator according to GLiClass ModernBERT contract.

    Formula: <<LABEL>>desc1<<LABEL>>desc2<<SEP>>text
    """
    label_prefix = "".join(f"{LABEL_MARKER}{label}" for label in label_descriptions)
    return f"{label_prefix}{SEP_MARKER}{text}"


def build_model_input(
    question: str,
    context: str,
    label_descriptions: Sequence[str],
) -> str:
    """Build the single canonical model input string across Python and JS runtimes.

    Formula: <<LABEL>>desc1<<LABEL>>desc2<<SEP>>Question: {question}\n\nContext:\n{context}
    """
    if question:
        text = INPUT_TEMPLATE.format(question=question, context=context)
    else:
        text = context
    return format_prompt(text, label_descriptions)


def format_query(
    context: str,
    query: Query,
) -> tuple[str, list[str], list[str]]:
    """Format query context and candidate labels.

    Returns:
        formatted_text: The input text combining context and question/proposition.
        candidate_labels: Natural-language label descriptions for GLiClass.
        candidate_ids: Corresponding stable option/level/noul IDs.
    """
    if query.kind == "choice":
        if len(query.options) > MAX_SUBSTANTIVE_CANDIDATES:
            raise CapacityError(len(query.options) + 1, MAX_SUPPORTED_CANDIDATES)
        formatted_text = f"Question: {query.question}\n\nContext:\n{context}"
        labels = [opt.description for opt in query.options] + [INSUFFICIENT_EVIDENCE_DESC]
        ids = [opt.id for opt in query.options] + [INSUFFICIENT_EVIDENCE_ID]
        return formatted_text, labels, ids

    if query.kind == "score":
        if len(query.levels) > MAX_SUBSTANTIVE_CANDIDATES:
            raise CapacityError(len(query.levels) + 1, MAX_SUPPORTED_CANDIDATES)
        formatted_text = f"Question: {query.question}\n\nContext:\n{context}"
        labels = [
            f"{lvl.description} (Value: {lvl.value})" for lvl in query.levels
        ] + [INSUFFICIENT_EVIDENCE_DESC]
        ids = [lvl.id for lvl in query.levels] + [INSUFFICIENT_EVIDENCE_ID]
        return formatted_text, labels, ids

    if query.kind == "noul":
        formatted_text = (
            f"Context:\n{context}\n\nEvaluate proposition: {query.proposition}"
        )
        labels = [
            f"true: {query.proposition}",
            f"false: not {query.proposition}",
            INSUFFICIENT_EVIDENCE_DESC,
        ]
        ids = ["true", "false", INSUFFICIENT_EVIDENCE_ID]
        return formatted_text, labels, ids

    raise TypeError(f"Unknown query kind: {query!r}")
