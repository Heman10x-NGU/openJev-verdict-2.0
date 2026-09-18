"""RLCD Decision Engine core package."""

from core.calibration import (
    ECEResult,
    TemperatureCalibrator,
    brier_score_loss,
    composite_loss,
    compute_ece,
    cross_entropy_loss,
)
from core.dag import (
    DAGCycleError,
    DAGMissingDependencyError,
    DAGNode,
    DecisionDAG,
)
from core.engine_causal import (
    CausalDecisionAdapter,
    TokenBoundaryError,
)
from core.engine_encoder import (
    DecisionEngine,
)
from core.primitives import (
    CHOICE_ADAPTER,
    DECISION_BATCH_RESULT_ADAPTER,
    DECISION_RESULT_ADAPTER,
    INSUFFICIENT_EVIDENCE_DESC,
    INSUFFICIENT_EVIDENCE_ID,
    NOUL_ADAPTER,
    QUERY_ADAPTER,
    SCORE_ADAPTER,
    Choice,
    ChoiceResult,
    DecisionBatchResult,
    DecisionResult,
    Level,
    Noul,
    NoulResult,
    Option,
    Query,
    Score,
    ScoreResult,
)
from core.tabular import (
    Cell,
    Table,
    TableCapacityError,
    TableCompactor,
)

__all__ = [
    # Decision Engine
    "DecisionEngine",
    # Primitives & Queries
    "Choice",
    "Option",
    "Score",
    "Level",
    "Noul",
    "Query",
    "INSUFFICIENT_EVIDENCE_ID",
    "INSUFFICIENT_EVIDENCE_DESC",
    # Results
    "ChoiceResult",
    "ScoreResult",
    "NoulResult",
    "DecisionResult",
    "DecisionBatchResult",
    # Adapters
    "CHOICE_ADAPTER",
    "SCORE_ADAPTER",
    "NOUL_ADAPTER",
    "QUERY_ADAPTER",
    "DECISION_RESULT_ADAPTER",
    "DECISION_BATCH_RESULT_ADAPTER",
    # Calibration & Losses
    "TemperatureCalibrator",
    "brier_score_loss",
    "cross_entropy_loss",
    "composite_loss",
    "compute_ece",
    "ECEResult",
    # Tabular
    "Table",
    "Cell",
    "TableCompactor",
    "TableCapacityError",
    # DAG
    "DecisionDAG",
    "DAGNode",
    "DAGCycleError",
    "DAGMissingDependencyError",
    # Causal
    "CausalDecisionAdapter",
    "TokenBoundaryError",
]
