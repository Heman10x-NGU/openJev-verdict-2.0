"""Unit tests for multi-stage topological DAG orchestration."""

import pytest

from core.dag import DAGCycleError, DAGMissingDependencyError, DecisionDAG
from core.primitives import (
    Choice,
    ChoiceResult,
    DecisionBatchResult,
    INSUFFICIENT_EVIDENCE_ID,
    Noul,
    Option,
    Score,
)


class MockEngine:
    """Mock decision engine simulating single-call evaluation."""

    def __init__(self, force_abstention: bool = False):
        self.force_abstention = force_abstention
        self.call_count = 0

    def evaluate(self, context: str, queries: list) -> DecisionBatchResult:
        self.call_count += 1
        results = []
        for q in queries:
            if self.force_abstention and q.id == "root_intent":
                results.append(
                    ChoiceResult(
                        id=q.id,
                        selected_id=INSUFFICIENT_EVIDENCE_ID,
                        selected_probability=0.92,
                        probabilities={
                            INSUFFICIENT_EVIDENCE_ID: 0.92,
                            q.options[0].id: 0.08,
                        },
                        is_abstention=True,
                    )
                )
            elif q.kind == "choice":
                results.append(
                    ChoiceResult(
                        id=q.id,
                        selected_id=q.options[0].id,
                        selected_probability=0.90,
                        probabilities={
                            q.options[0].id: 0.90,
                            INSUFFICIENT_EVIDENCE_ID: 0.10,
                        },
                        is_abstention=False,
                    )
                )
            elif q.kind == "noul":
                results.append(
                    ChoiceResult(
                        id=q.id,
                        selected_id="true",
                        selected_probability=0.95,
                        probabilities={"true": 0.95, INSUFFICIENT_EVIDENCE_ID: 0.05},
                        is_abstention=False,
                    )
                )
        return DecisionBatchResult(
            results=tuple(results),
            forward_call_count=1,
            total_latency_ms=5.0,
            execution_mode="single_call",
        )


def test_dag_cycle_detection() -> None:
    dag = DecisionDAG()
    q1 = Choice(
        id="a",
        question="Q1",
        options=[Option(id="1", description="1"), Option(id="2", description="2")],
    )
    q2 = Choice(
        id="b",
        question="Q2",
        options=[Option(id="1", description="1"), Option(id="2", description="2")],
    )

    dag.add_node("a", q1, depends_on=["b"])
    dag.add_node("b", q2, depends_on=["a"])

    with pytest.raises(DAGCycleError, match="Cyclical dependency"):
        dag.get_topological_stages()


def test_dag_missing_dependency() -> None:
    dag = DecisionDAG()
    q1 = Choice(
        id="a",
        question="Q1",
        options=[Option(id="1", description="1"), Option(id="2", description="2")],
    )
    dag.add_node("a", q1, depends_on=["non_existent_parent"])

    with pytest.raises(DAGMissingDependencyError, match="missing parent"):
        dag.get_topological_stages()


def test_dag_multi_stage_two_pass_execution() -> None:
    dag = DecisionDAG()
    root_q = Choice(
        id="root_intent",
        question="What is the user's primary intent?",
        options=[
            Option(id="refund", description="Customer wants a refund"),
            Option(id="exchange", description="Customer wants an exchange"),
        ],
    )
    dag.add_node("root_intent", root_q)

    # Dynamic child query conditioned on parent result
    def child_factory(parent_results):
        intent = parent_results["root_intent"].selected_id
        return Noul(
            id="eligibility",
            proposition=f"The customer is eligible for action {intent}.",
            semantics="conditional_on_sufficient_evidence_v2",
        )

    dag.add_node("eligibility", child_factory, depends_on=["root_intent"])

    stages = dag.get_topological_stages()
    assert stages == [["root_intent"], ["eligibility"]]

    mock_engine = MockEngine(force_abstention=False)
    batch_res = dag.execute(mock_engine, context="Customer invoice details...")

    assert batch_res.forward_call_count == 2
    assert batch_res.execution_mode == "orchestrated"
    assert len(batch_res.results) == 2
    assert batch_res.results[0].selected_id == "refund"
    assert not batch_res.results[1].is_abstention


def test_dag_blocks_child_on_parent_abstention() -> None:
    dag = DecisionDAG()
    root_q = Choice(
        id="root_intent",
        question="Intent?",
        options=[
            Option(id="refund", description="Refund"),
            Option(id="exchange", description="Exchange"),
        ],
    )
    dag.add_node("root_intent", root_q)

    child_q = Noul(
        id="eligibility",
        proposition="Eligible?",
        semantics="conditional_on_sufficient_evidence_v2",
    )
    dag.add_node("eligibility", child_q, depends_on=["root_intent"])

    mock_engine = MockEngine(force_abstention=True)
    batch_res = dag.execute(mock_engine, context="Unclear complaint...")

    # Root executed, but child was blocked because parent abstained
    assert batch_res.forward_call_count == 1
    assert batch_res.results[0].is_abstention
    assert batch_res.results[1].is_abstention
    assert batch_res.results[1].calibration_status == "blocked_by_parent_abstention"
