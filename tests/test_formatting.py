"""Unit tests for core/formatting.py prompt construction and capacity guards."""

import pytest

from core.formatting import (
    CapacityError,
    MAX_SUBSTANTIVE_CANDIDATES,
    MAX_SUPPORTED_CANDIDATES,
    format_prompt,
    format_query,
)
from core.primitives import Choice, Option, Score, Level, Noul


def test_format_prompt_structure() -> None:
    prompt = format_prompt("What is the refund amount?", ["billing", "tech", "insufficient evidence"])
    expected = "<<LABEL>>billing<<LABEL>>tech<<LABEL>>insufficient evidence<<SEP>>What is the refund amount?"
    assert prompt == expected


def test_format_query_choice() -> None:
    q = Choice(
        id="q1",
        question="Select action",
        options=[
            Option(id="refund", description="Issue full refund"),
            Option(id="cancel", description="Cancel active subscription"),
        ],
    )
    text, labels, ids = format_query("Customer requested refund on item #42", q)
    assert text == "Question: Select action\n\nContext:\nCustomer requested refund on item #42"
    assert labels == ["Issue full refund", "Cancel active subscription", "insufficient evidence"]
    assert ids == ["refund", "cancel", "__insufficient_evidence__"]


def test_choice_capacity_guard() -> None:
    # 24 options allowed (24 + 1 abstention = 25 total)
    valid_options = [Option(id=f"opt_{i}", description=f"Option {i}") for i in range(24)]
    choice = Choice(id="q", question="test", options=valid_options)
    assert len(choice.options) == 24

    # 25 options must raise ValidationError because total capacity is 25 including abstention
    with pytest.raises(Exception):
        invalid_options = [Option(id=f"opt_{i}", description=f"Option {i}") for i in range(25)]
        Choice(id="q", question="test", options=invalid_options)


def test_noul_semantics_required() -> None:
    # Without semantics field, Noul must fail validation
    with pytest.raises(Exception):
        Noul(id="n1", proposition="Server is healthy")  # type: ignore[call-arg]

    # With semantics, it succeeds
    n = Noul(id="n1", proposition="Server is healthy", semantics="conditional_on_sufficient_evidence_v2")
    assert n.semantics == "conditional_on_sufficient_evidence_v2"
