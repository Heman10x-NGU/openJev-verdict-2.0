"""Typed decision primitives and schemas for non-autoregressive inference.

This module defines Pydantic v2 schemas for Choice, Score, and Noul queries,
along with their corresponding decision results and cached TypeAdapters.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Sequence, TypeAlias

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    FiniteFloat,
    TypeAdapter,
    field_validator,
    model_validator,
)

INSUFFICIENT_EVIDENCE_ID = "__insufficient_evidence__"
INSUFFICIENT_EVIDENCE_DESC = "insufficient evidence"

TRUE_OUTCOME_ID = "true"
FALSE_OUTCOME_ID = "false"


class FrozenModel(BaseModel):
    """Immutable model with extra fields forbidden."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


class Option(FrozenModel):
    """A categorical choice option with an identifier and semantic description."""

    id: str = Field(min_length=1)
    description: str = Field(min_length=1)


class Level(Option):
    """An ordinal rubric level with an explicit numerical scalar value."""

    value: FiniteFloat


class Choice(FrozenModel):
    """A categorical single-choice query with mutually exclusive options."""

    kind: Literal["choice"] = "choice"
    id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    options: tuple[Option, ...] = Field(min_length=2, max_length=24)

    @field_validator("options", mode="before")
    @classmethod
    def _coerce_options(cls, v: Any) -> Any:
        if isinstance(v, (list, tuple)):
            coerced: list[Option] = []
            for item in v:
                if isinstance(item, dict):
                    coerced.append(Option(**item))
                elif isinstance(item, Option):
                    coerced.append(item)
                else:
                    raise TypeError(f"Invalid option item: {item!r}")
            return tuple(coerced)
        return v

    @model_validator(mode="after")
    def _validate_choice(self) -> Choice:
        ids: set[str] = set()
        for opt in self.options:
            if opt.id == INSUFFICIENT_EVIDENCE_ID or opt.id == "insufficient_evidence":
                raise ValueError(
                    f"Option ID {opt.id!r} is reserved for explicit abstention."
                )
            if opt.id in ids:
                raise ValueError(f"Duplicate option ID detected: {opt.id!r}")
            ids.add(opt.id)
        return self


class Score(FrozenModel):
    """An ordinal scoring query with ordered numerical rubric levels."""

    kind: Literal["score"] = "score"
    id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    levels: tuple[Level, ...] = Field(min_length=2, max_length=24)

    @field_validator("levels", mode="before")
    @classmethod
    def _coerce_levels(cls, v: Any) -> Any:
        if isinstance(v, (list, tuple)):
            coerced: list[Level] = []
            for item in v:
                if isinstance(item, dict):
                    coerced.append(Level(**item))
                elif isinstance(item, Level):
                    coerced.append(item)
                else:
                    raise TypeError(f"Invalid level item: {item!r}")
            return tuple(coerced)
        return v

    @model_validator(mode="after")
    def _validate_score(self) -> Score:
        ids: set[str] = set()
        prev_val: float | None = None
        for lvl in self.levels:
            if lvl.id == INSUFFICIENT_EVIDENCE_ID or lvl.id == "insufficient_evidence":
                raise ValueError(
                    f"Level ID {lvl.id!r} is reserved for explicit abstention."
                )
            if lvl.id in ids:
                raise ValueError(f"Duplicate level ID detected: {lvl.id!r}")
            ids.add(lvl.id)
            if prev_val is not None and lvl.value <= prev_val:
                raise ValueError(
                    f"Score levels must have strictly increasing numerical values: "
                    f"level {lvl.id!r} with value {lvl.value} is not greater than {prev_val}."
                )
            prev_val = lvl.value
        return self


class Noul(FrozenModel):
    """A bounded semantic proposition returning true, false, or abstention.

    Under rlcd-evidence-v2 semantics, evaluates evidence across three outcomes:
    true, false, and insufficient evidence. Semantics field is strictly required.
    """

    kind: Literal["noul"] = "noul"
    id: str = Field(min_length=1)
    proposition: str = Field(min_length=1)
    semantics: Literal["conditional_on_sufficient_evidence_v2"]


Query: TypeAlias = Annotated[Choice | Score | Noul, Field(discriminator="kind")]


class ChoiceResult(FrozenModel):
    """Inference result for a Choice query."""

    kind: Literal["choice"] = "choice"
    id: str
    selected_id: str
    selected_probability: FiniteFloat
    probabilities: dict[str, FiniteFloat]
    is_abstention: bool
    concentration: FiniteFloat | None = None
    model_id: str = ""
    calibration_status: str = "uncalibrated"
    latency_ms: FiniteFloat = 0.0

    @model_validator(mode="after")
    def _validate_choice_result(self) -> ChoiceResult:
        if self.latency_ms < 0.0:
            raise ValueError(f"latency_ms cannot be negative: {self.latency_ms}")
        if self.selected_id not in self.probabilities:
            raise ValueError(f"selected_id {self.selected_id!r} not in probabilities dictionary.")
        prob_sum = sum(self.probabilities.values())
        if abs(prob_sum - 1.0) > 1e-3:
            raise ValueError(f"Probabilities must sum to 1.0 (got {prob_sum})")
        # Ensure selected_id corresponds to maximum probability
        max_id = max(self.probabilities, key=self.probabilities.get)  # type: ignore[arg-type]
        if abs(self.probabilities[self.selected_id] - self.probabilities[max_id]) > 1e-5:
            raise ValueError(
                f"selected_id {self.selected_id!r} does not have highest probability ({self.probabilities[self.selected_id]} vs max {self.probabilities[max_id]})"
            )
        return self


class ScoreResult(FrozenModel):
    """Inference result for a Score query."""

    kind: Literal["score"] = "score"
    id: str
    selected_level_id: str
    selected_value: FiniteFloat | None
    expected_score: FiniteFloat | None
    probabilities: dict[str, FiniteFloat]
    is_abstention: bool
    abstention_probability: FiniteFloat
    model_id: str = ""
    calibration_status: str = "uncalibrated"
    latency_ms: FiniteFloat = 0.0

    @model_validator(mode="after")
    def _validate_score_result(self) -> ScoreResult:
        if self.latency_ms < 0.0:
            raise ValueError(f"latency_ms cannot be negative: {self.latency_ms}")
        if self.selected_level_id not in self.probabilities:
            raise ValueError(f"selected_level_id {self.selected_level_id!r} not in probabilities dictionary.")
        prob_sum = sum(self.probabilities.values())
        if abs(prob_sum - 1.0) > 1e-3:
            raise ValueError(f"Probabilities must sum to 1.0 (got {prob_sum})")
        return self


class NoulResult(FrozenModel):
    """Inference result for a Noul proposition query."""

    kind: Literal["noul"] = "noul"
    id: str
    selected_outcome: Literal["true", "false", "__insufficient_evidence__"]
    p_true_given_sufficient_evidence: FiniteFloat | None
    p_insufficient_evidence: FiniteFloat
    probabilities: dict[str, FiniteFloat]
    is_abstention: bool
    model_id: str = ""
    calibration_status: str = "uncalibrated"
    latency_ms: FiniteFloat = 0.0

    @model_validator(mode="after")
    def _validate_noul_result(self) -> NoulResult:
        if self.latency_ms < 0.0:
            raise ValueError(f"latency_ms cannot be negative: {self.latency_ms}")
        if self.selected_outcome not in self.probabilities:
            raise ValueError(f"selected_outcome {self.selected_outcome!r} not in probabilities dictionary.")
        prob_sum = sum(self.probabilities.values())
        if abs(prob_sum - 1.0) > 1e-3:
            raise ValueError(f"Probabilities must sum to 1.0 (got {prob_sum})")
        return self


DecisionResult: TypeAlias = Annotated[
    ChoiceResult | ScoreResult | NoulResult, Field(discriminator="kind")
]


class DecisionBatchResult(FrozenModel):
    """Batch result containing evaluated decisions and execution telemetry."""

    results: tuple[DecisionResult, ...]
    forward_call_count: int
    total_latency_ms: float
    execution_mode: Literal["single_call", "orchestrated"] = "single_call"

    @field_validator("results", mode="before")
    @classmethod
    def _coerce_results(cls, v: Any) -> Any:
        if isinstance(v, (list, tuple)):
            return tuple(v)
        return v


CHOICE_ADAPTER: TypeAdapter[Choice] = TypeAdapter(Choice)
SCORE_ADAPTER: TypeAdapter[Score] = TypeAdapter(Score)
NOUL_ADAPTER: TypeAdapter[Noul] = TypeAdapter(Noul)
QUERY_ADAPTER: TypeAdapter[Query] = TypeAdapter(Query)
DECISION_RESULT_ADAPTER: TypeAdapter[DecisionResult] = TypeAdapter(DecisionResult)
DECISION_BATCH_RESULT_ADAPTER: TypeAdapter[DecisionBatchResult] = TypeAdapter(
    DecisionBatchResult
)
