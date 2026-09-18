"""Unit tests for typed decision primitives."""

import pytest
from pydantic import ValidationError

from core.primitives import (
    CHOICE_ADAPTER,
    DECISION_BATCH_RESULT_ADAPTER,
    DECISION_RESULT_ADAPTER,
    INSUFFICIENT_EVIDENCE_ID,
    NOUL_ADAPTER,
    SCORE_ADAPTER,
    Choice,
    ChoiceResult,
    DecisionBatchResult,
    Level,
    Noul,
    NoulResult,
    Option,
    Score,
    ScoreResult,
)


def test_choice_valid_and_serialization() -> None:
    choice = Choice(
        id="q1",
        question="Select the department",
        options=[
            Option(id="billing", description="Billing and invoice issues"),
            Option(id="tech", description="Technical support"),
        ],
    )
    assert choice.kind == "choice"
    assert len(choice.options) == 2

    # Round-trip through TypeAdapter
    json_bytes = CHOICE_ADAPTER.dump_json(choice)
    loaded = CHOICE_ADAPTER.validate_json(json_bytes)
    assert loaded == choice


def test_choice_rejects_reserved_abstention_id() -> None:
    with pytest.raises(ValidationError, match="reserved for explicit abstention"):
        Choice(
            id="q1",
            question="Select the department",
            options=[
                Option(id=INSUFFICIENT_EVIDENCE_ID, description="Reserved"),
                Option(id="tech", description="Technical support"),
            ],
        )


def test_choice_rejects_duplicate_options() -> None:
    with pytest.raises(ValidationError, match="Duplicate option ID"):
        Choice(
            id="q1",
            question="Select department",
            options=[
                Option(id="support", description="Support 1"),
                Option(id="support", description="Support 2"),
            ],
        )


def test_score_strictly_increasing_validation() -> None:
    # Valid strictly increasing
    score = Score(
        id="s1",
        question="Rate severity",
        levels=[
            Level(id="low", description="Minor inconvenience", value=1.0),
            Level(id="med", description="Degraded performance", value=2.0),
            Level(id="high", description="System outage", value=3.0),
        ],
    )
    assert len(score.levels) == 3

    # Non-increasing values must fail
    with pytest.raises(ValidationError, match="strictly increasing"):
        Score(
            id="s2",
            question="Rate urgency",
            levels=[
                Level(id="p1", description="High", value=5.0),
                Level(id="p2", description="Low", value=1.0),
            ],
        )


def test_noul_semantics_and_serialization() -> None:
    # Omitted semantics must fail
    with pytest.raises(ValidationError):
        Noul(
            id="n0",
            proposition="Omitted semantics should raise error.",
        )  # type: ignore[call-arg]

    noul = Noul(
        id="n1",
        proposition="The refund request complies with the 30-day window policy.",
        semantics="conditional_on_sufficient_evidence_v2",
    )
    assert noul.kind == "noul"
    assert noul.semantics == "conditional_on_sufficient_evidence_v2"

    json_data = NOUL_ADAPTER.dump_json(noul)
    reconstructed = NOUL_ADAPTER.validate_json(json_data)
    assert reconstructed == noul


def test_decision_results_and_batch_adapter() -> None:
    c_res = ChoiceResult(
        id="q1",
        selected_id="billing",
        selected_probability=0.85,
        probabilities={"billing": 0.85, "tech": 0.10, INSUFFICIENT_EVIDENCE_ID: 0.05},
        is_abstention=False,
    )
    s_res = ScoreResult(
        id="s1",
        selected_level_id="high",
        selected_value=3.0,
        expected_score=2.8,
        probabilities={"low": 0.05, "med": 0.15, "high": 0.75, INSUFFICIENT_EVIDENCE_ID: 0.05},
        is_abstention=False,
        abstention_probability=0.05,
    )
    n_res = NoulResult(
        id="n1",
        selected_outcome="true",
        p_true_given_sufficient_evidence=0.90,
        p_insufficient_evidence=0.10,
        probabilities={"true": 0.81, "false": 0.09, INSUFFICIENT_EVIDENCE_ID: 0.10},
        is_abstention=False,
    )

    batch = DecisionBatchResult(
        results=[c_res, s_res, n_res],
        forward_call_count=1,
        total_latency_ms=12.5,
        execution_mode="single_call",
    )

    dumped = DECISION_BATCH_RESULT_ADAPTER.dump_json(batch)
    loaded_batch = DECISION_BATCH_RESULT_ADAPTER.validate_json(dumped)
    assert loaded_batch == batch
    assert len(loaded_batch.results) == 3
    assert loaded_batch.results[0].kind == "choice"
    assert loaded_batch.results[1].kind == "score"
    assert loaded_batch.results[2].kind == "noul"
