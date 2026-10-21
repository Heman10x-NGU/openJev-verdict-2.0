"""Offline regression tests for the decision engine's backend output contract."""

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from core.calibration import TemperatureCalibrator
from core.engine_encoder import DecisionEngine
from core.primitives import (
    Choice,
    INSUFFICIENT_EVIDENCE_ID,
    Level,
    Noul,
    Option,
    Score,
)


class FixedLogitModel(torch.nn.Module):
    def __init__(self, logits: torch.Tensor) -> None:
        super().__init__()
        self.logits = logits
        self.calls = 0

    def forward(self, **inputs: Any) -> SimpleNamespace:
        self.calls += 1
        return SimpleNamespace(logits=self.logits)


class TokenBatch(dict[str, torch.Tensor]):
    def to(self, device: str) -> "TokenBatch":
        return TokenBatch({key: value.to(device) for key, value in self.items()})


class FixedTokenizer:
    def __call__(self, prompts: list[str], **kwargs: Any) -> TokenBatch:
        ids = torch.ones((len(prompts), 1), dtype=torch.long)
        return TokenBatch(input_ids=ids, attention_mask=ids)


def make_engine(logits: torch.Tensor, backend: str) -> DecisionEngine:
    engine = DecisionEngine(
        model=FixedLogitModel(logits),
        tokenizer=FixedTokenizer(),
        calibrator=TemperatureCalibrator(),
    )
    if backend == "onnx":
        engine.ort_session = SimpleNamespace(
            run=lambda *_: [logits.numpy()]
        )
    return engine


QUERIES = [
    Choice(
        id="route",
        question="Choose a team",
        options=[
            Option(id="billing", description="Billing"),
            Option(id="support", description="Support"),
        ],
    ),
    Score(
        id="severity",
        question="Rate severity",
        levels=[
            Level(id="low", description="Low", value=1.0),
            Level(id="high", description="High", value=2.0),
        ],
    ),
    Noul(
        id="refund",
        proposition="The customer requests a refund.",
        semantics="conditional_on_sufficient_evidence_v2",
    ),
]


@pytest.mark.parametrize("backend", ["pytorch", "onnx"])
@pytest.mark.parametrize("query", QUERIES, ids=lambda query: query.kind)
def test_missing_abstention_logit_is_rejected(query: Any, backend: str) -> None:
    engine = make_engine(torch.tensor([[3.0, 0.5]]), backend)
    with pytest.raises(ValueError, match="candidate logits"):
        engine.evaluate("A customer needs help.", [query])


@pytest.mark.parametrize("backend", ["pytorch", "onnx"])
@pytest.mark.parametrize(
    "shape, message",
    [
        ((), "two-dimensional"),
        ((3,), "two-dimensional"),
        ((1, 1, 3), "two-dimensional"),
        ((0, 3), "batch"),
        ((2, 3), "batch"),
    ],
)
def test_malformed_output_shape_is_rejected(
    shape: tuple[int, ...], message: str, backend: str
) -> None:
    engine = make_engine(torch.zeros(shape), backend)
    with pytest.raises(ValueError, match=message):
        engine.evaluate("A customer needs help.", [QUERIES[0]])


@pytest.mark.parametrize("backend", ["pytorch", "onnx"])
@pytest.mark.parametrize(
    "query, candidate_ids",
    [
        (QUERIES[0], ["billing", "support", INSUFFICIENT_EVIDENCE_ID]),
        (QUERIES[1], ["low", "high", INSUFFICIENT_EVIDENCE_ID]),
        (QUERIES[2], ["true", "false", INSUFFICIENT_EVIDENCE_ID]),
    ],
    ids=["choice", "score", "noul"],
)
def test_valid_output_preserves_candidates_and_ignores_padding(
    query: Any, candidate_ids: list[str], backend: str
) -> None:
    engine = make_engine(torch.tensor([[2.0, 1.0, 0.0, 1000.0]]), backend)
    result = engine.evaluate("A customer needs help.", [query])
    assert list(result.results[0].probabilities) == candidate_ids
    assert list(result.results[0].probabilities.values()) == pytest.approx(
        [0.66524096, 0.24472847, 0.09003057]
    )
    assert result.forward_call_count == 1


@pytest.mark.parametrize("backend", ["pytorch", "onnx"])
def test_mixed_candidate_counts_preserve_abstention(backend: str) -> None:
    choice = Choice(
        id="route",
        question="Choose a team",
        options=[
            Option(id="billing", description="Billing"),
            Option(id="support", description="Support"),
            Option(id="sales", description="Sales"),
        ],
    )
    engine = make_engine(
        torch.tensor([[1.0, 2.0, 3.0, 4.0], [3.0, 2.0, 1.0, 1000.0]]),
        backend,
    )
    result = engine.evaluate("A customer needs help.", [choice, QUERIES[2]])
    assert len(result.results) == 2
    assert result.results[0].selected_id == INSUFFICIENT_EVIDENCE_ID
    assert result.results[0].is_abstention
    assert result.results[1].selected_outcome == "true"
    assert list(result.results[1].probabilities) == [
        "true", "false", INSUFFICIENT_EVIDENCE_ID
    ]
