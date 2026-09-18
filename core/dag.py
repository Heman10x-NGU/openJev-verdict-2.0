"""Topological DAG orchestrator for multi-stage dependent decision queries.

Schedules dependent queries into topological stages. Independent root queries
execute concurrently in pass one; dependent downstream queries execute in
pass two with parent decision results injected into context. Automatically
halts downstream execution when an upstream prerequisite abstains.
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any, Callable, Sequence

from core.primitives import (
    ChoiceResult,
    DecisionBatchResult,
    DecisionResult,
    INSUFFICIENT_EVIDENCE_ID,
    NoulResult,
    Query,
    ScoreResult,
)


class DAGCycleError(ValueError):
    """Raised when a cyclical dependency is detected in the decision DAG."""


class DAGMissingDependencyError(ValueError):
    """Raised when a node references an unknown upstream prerequisite."""


class DAGNode:
    """A node in the decision DAG representing a typed query or query factory."""

    def __init__(
        self,
        node_id: str,
        query_or_factory: Query | Callable[[dict[str, DecisionResult]], Query],
        depends_on: Sequence[str] = (),
    ):
        self.node_id = node_id
        self.query_or_factory = query_or_factory
        self.depends_on = tuple(depends_on)

    def resolve_query(self, parent_results: dict[str, DecisionResult]) -> Query:
        """Resolve query instance, evaluating factory with parent results if needed."""
        if callable(self.query_or_factory):
            return self.query_or_factory(parent_results)
        return self.query_or_factory


class DecisionDAG:
    """Directed Acyclic Graph orchestrating multi-pass dependent decision queries."""

    def __init__(self) -> None:
        self.nodes: dict[str, DAGNode] = {}

    def add_node(
        self,
        node_id: str,
        query_or_factory: Query | Callable[[dict[str, DecisionResult]], Query],
        depends_on: Sequence[str] = (),
    ) -> DecisionDAG:
        """Add a decision node to the graph."""
        if node_id in self.nodes:
            raise ValueError(f"Node {node_id!r} already exists in DAG.")
        self.nodes[node_id] = DAGNode(
            node_id=node_id,
            query_or_factory=query_or_factory,
            depends_on=depends_on,
        )
        return self

    def get_topological_stages(self) -> list[list[str]]:
        """Compute topological execution stages using Kahn's algorithm."""
        in_degree: dict[str, int] = {node_id: 0 for node_id in self.nodes}
        dependents: dict[str, list[str]] = defaultdict(list)

        for node_id, node in self.nodes.items():
            for parent_id in node.depends_on:
                if parent_id not in self.nodes:
                    raise DAGMissingDependencyError(
                        f"Node {node_id!r} depends on missing parent {parent_id!r}."
                    )
                dependents[parent_id].append(node_id)
                in_degree[node_id] += 1

        queue: deque[str] = deque([nid for nid, deg in in_degree.items() if deg == 0])
        stages: list[list[str]] = []
        visited_count = 0

        while queue:
            current_stage: list[str] = []
            next_queue: deque[str] = deque()
            while queue:
                u = queue.popleft()
                current_stage.append(u)
                visited_count += 1
                for v in dependents[u]:
                    in_degree[v] -= 1
                    if in_degree[v] == 0:
                        next_queue.append(v)
            stages.append(current_stage)
            queue = next_queue

        if visited_count != len(self.nodes):
            raise DAGCycleError("Cyclical dependency detected in decision DAG.")

        return stages

    def execute(
        self,
        engine: Any,
        context: str,
    ) -> DecisionBatchResult:
        """Execute decision DAG through batched topological stages.

        Args:
            engine: Engine instance implementing evaluate(context, queries).
            context: Shared textual context for decisions.

        Returns:
            DecisionBatchResult aggregating results across all execution stages.
        """
        stages = self.get_topological_stages()
        all_results: dict[str, DecisionResult] = {}
        total_forward_calls = 0
        total_latency_ms = 0.0

        for stage in stages:
            stage_queries: list[Query] = []
            stage_node_ids: list[str] = []

            for node_id in stage:
                node = self.nodes[node_id]
                # Check if any parent abstained
                parent_abstained = False
                for p_id in node.depends_on:
                    p_res = all_results.get(p_id)
                    if p_res is not None and p_res.is_abstention:
                        parent_abstained = True
                        break

                if parent_abstained:
                    # Parent abstained: block execution of this dependent node
                    dummy_query = node.resolve_query(all_results)
                    blocked_result: DecisionResult
                    if dummy_query.kind == "choice":
                        blocked_result = ChoiceResult(
                            id=node_id,
                            selected_id=INSUFFICIENT_EVIDENCE_ID,
                            selected_probability=1.0,
                            probabilities={INSUFFICIENT_EVIDENCE_ID: 1.0},
                            is_abstention=True,
                            model_id="policy_orchestrator",
                            calibration_status="blocked_by_parent_abstention",
                            latency_ms=0.0,
                        )
                    elif dummy_query.kind == "score":
                        blocked_result = ScoreResult(
                            id=node_id,
                            selected_level_id=INSUFFICIENT_EVIDENCE_ID,
                            selected_value=None,
                            expected_score=None,
                            probabilities={INSUFFICIENT_EVIDENCE_ID: 1.0},
                            is_abstention=True,
                            abstention_probability=1.0,
                            model_id="policy_orchestrator",
                            calibration_status="blocked_by_parent_abstention",
                            latency_ms=0.0,
                        )
                    else:
                        blocked_result = NoulResult(
                            id=node_id,
                            selected_outcome=INSUFFICIENT_EVIDENCE_ID,
                            p_true_given_sufficient_evidence=None,
                            p_insufficient_evidence=1.0,
                            probabilities={INSUFFICIENT_EVIDENCE_ID: 1.0},
                            is_abstention=True,
                            model_id="policy_orchestrator",
                            calibration_status="blocked_by_parent_abstention",
                            latency_ms=0.0,
                        )
                    all_results[node_id] = blocked_result
                else:
                    query = node.resolve_query(all_results)
                    stage_queries.append(query)
                    stage_node_ids.append(node_id)

            if stage_queries:
                batch_res: DecisionBatchResult = engine.evaluate(
                    context=context,
                    queries=stage_queries,
                )
                total_forward_calls += batch_res.forward_call_count
                total_latency_ms += batch_res.total_latency_ms
                for nid, r in zip(stage_node_ids, batch_res.results):
                    all_results[nid] = r

        # Preserve original DAG insertion order
        ordered_results = [all_results[nid] for nid in self.nodes]
        return DecisionBatchResult(
            results=tuple(ordered_results),
            forward_call_count=total_forward_calls,
            total_latency_ms=total_latency_ms,
            execution_mode="orchestrated",
        )
