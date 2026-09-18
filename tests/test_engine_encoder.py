"""Unit and integration tests for primary ModernBERT / GLiClass DecisionEngine."""

import pytest

from core.engine_encoder import DecisionEngine, _compute_concentration
from core.primitives import (
    Choice,
    INSUFFICIENT_EVIDENCE_ID,
    Level,
    Noul,
    Option,
    Score,
)


def test_compute_concentration() -> None:
    # Uniform distribution: concentration should be 0.0
    uniform_p = [0.25, 0.25, 0.25, 0.25]
    assert pytest.approx(_compute_concentration(uniform_p), abs=1e-5) == 0.0

    # Point mass distribution: concentration should be 1.0
    point_mass = [1.0, 0.0, 0.0, 0.0]
    assert pytest.approx(_compute_concentration(point_mass), abs=1e-5) == 1.0


def test_decision_engine_mock_batch_execution() -> None:
    engine = DecisionEngine(model="mock")

    q1 = Choice(
        id="q1",
        question="Select category",
        options=[
            Option(id="billing", description="Billing question"),
            Option(id="tech", description="Technical issue"),
        ],
    )
    q2 = Score(
        id="q2",
        question="Rate severity",
        levels=[
            Level(id="p1", description="Low", value=1.0),
            Level(id="p2", description="High", value=5.0),
        ],
    )
    q3 = Noul(
        id="q3",
        proposition="User is requesting a refund.",
        semantics="conditional_on_sufficient_evidence_v2",
    )

    batch_res = engine.evaluate("Customer wants a refund on order #1234", [q1, q2, q3])
    assert batch_res.forward_call_count == 1
    assert batch_res.execution_mode == "single_call"
    assert len(batch_res.results) == 3

    # Check Choice result
    c_res = batch_res.results[0]
    assert c_res.id == "q1"
    assert c_res.selected_id == "billing"
    assert not c_res.is_abstention
    assert c_res.concentration is not None

    # Check Score result
    s_res = batch_res.results[1]
    assert s_res.id == "q2"
    assert s_res.selected_level_id == "p1"
    assert s_res.selected_value == 1.0
    assert s_res.expected_score is not None

    # Check Noul result
    n_res = batch_res.results[2]
    assert n_res.id == "q3"
    assert n_res.selected_outcome in ("true", "false", INSUFFICIENT_EVIDENCE_ID)


def test_decision_engine_injected_model_and_tokenizer() -> None:
    from gliclass import GLiClassModel
    from transformers import AutoTokenizer

    model_name = "knowledgator/gliclass-modern-base-v2.0"
    model = GLiClassModel.from_pretrained(model_name).cpu().eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    engine = DecisionEngine(model=model, tokenizer=tokenizer, device="cpu")
    q = Choice(
        id="c1",
        question="Select category",
        options=[
            Option(id="card", description="Card replacement"),
            Option(id="loan", description="Mortgage loan query"),
        ],
    )
    res = engine.evaluate("I need a new debit card sent to my home address", [q])
    assert len(res.results) == 1
    assert res.results[0].selected_id == "card"


def test_decision_engine_real_gliclass_inference() -> None:
    # Test real checkpoint inference on CPU
    engine = DecisionEngine(model_name_or_path="knowledgator/gliclass-modern-base-v2.0", device="cpu")

    context = (
        "Hello, I was double-charged on my credit card statement for invoice INV-9021. "
        "Please reverse the duplicate charge of $49.00."
    )

    q_choice = Choice(
        id="intent",
        question="What is the primary customer request?",
        options=[
            Option(id="billing_dispute", description="Billing dispute or duplicate charge refund"),
            Option(id="password_reset", description="Account login or password reset request"),
            Option(id="shipping_status", description="Package delivery or shipping tracking"),
        ],
    )

    q_noul = Noul(
        id="has_invoice_id",
        proposition="The customer explicitly provides an invoice identifier in the context.",
        semantics="conditional_on_sufficient_evidence_v2",
    )

    batch_res = engine.evaluate(context, [q_choice, q_noul])
    assert batch_res.forward_call_count == 1
    assert len(batch_res.results) == 2

    # Intent should clearly be billing_dispute
    intent_res = batch_res.results[0]
    assert intent_res.selected_id == "billing_dispute"
    assert intent_res.selected_probability > 0.70
    assert not intent_res.is_abstention

    # Proposition evaluation
    noul_res = batch_res.results[1]
    assert noul_res.selected_outcome == "true"
    assert not noul_res.is_abstention
