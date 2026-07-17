"""Core grading contracts, rubric schemas, and score values."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import math
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import (
    Any,
    Generic,
    Literal,
    Self,
    TypeAlias,
    TypeVar,
    cast,
    final,
    overload,
)

from gymnasium.spaces import Space
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    field_validator,
    model_validator,
)


InputT = TypeVar("InputT", contravariant=True)
CallableInputT = TypeVar("CallableInputT")
RubricInputT = TypeVar("RubricInputT")


def _strip_required(value: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError("must not be empty")
    return value


def _strip_optional(value: str | None) -> str | None:
    if value is None:
        return None
    return _strip_required(value)


class Level(BaseModel):
    """A named performance level available for one rubric criterion.

    Scores are normalized within a criterion. Criterion weights express the
    relative importance of the criterion when a grader explicitly aggregates
    component scores.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    description: str
    score: float = Field(ge=0.0, le=1.0)
    label: str | None = None

    _normalize_required = field_validator("id", "description")(_strip_required)
    _normalize_optional = field_validator("label")(_strip_optional)

    @field_validator("score")
    @classmethod
    def _finite_score(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("level score must be finite")
        return value


class Criterion(BaseModel):
    """One independently assessable requirement in a rubric.

    ``levels`` optionally define discrete classroom-style performance bands.
    An empty level collection leaves the criterion open to continuous scoring.
    Criteria are intentionally flat: grouping belongs in ``category`` or
    ``metadata`` while each criterion remains independently gradeable.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    description: str
    weight: float = Field(default=1.0, gt=0.0)
    title: str | None = None
    levels: tuple[Level, ...] = ()
    category: str | None = None
    references: tuple[str, ...] = ()
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    _normalize_required = field_validator("id", "description")(_strip_required)
    _normalize_optional = field_validator("title", "category")(_strip_optional)

    @field_validator("references")
    @classmethod
    def _non_empty_references(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        references = tuple(value.strip() for value in values)
        if any(not value for value in references):
            raise ValueError("references must not contain empty values")
        return references

    @field_validator("weight")
    @classmethod
    def _finite_weight(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("criterion weight must be finite")
        return value

    @model_validator(mode="after")
    def _unique_levels(self) -> Self:
        ids = [level.id for level in self.levels]
        if len(ids) != len(set(ids)):
            raise ValueError("level ids must be unique within a criterion")
        scores = [level.score for level in self.levels]
        if len(scores) != len(set(scores)):
            raise ValueError("level scores must be unique within a criterion")
        return self

    def level(self, level_id: str) -> Level:
        """Return a performance level by identifier."""

        for level in self.levels:
            if level.id == level_id:
                return level
        raise KeyError(level_id)


class Rubric(BaseModel):
    """A portable, versioned collection of grading criteria."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1"] = "1"
    id: str | None = None
    version: str | None = None
    title: str | None = None
    description: str | None = None
    instructions: str | None = None
    criteria: tuple[Criterion, ...] = Field(min_length=1)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    _normalize_optional = field_validator(
        "id",
        "version",
        "title",
        "description",
        "instructions",
    )(_strip_optional)

    @model_validator(mode="after")
    def _unique_criterion_ids(self) -> Self:
        ids = [criterion.id for criterion in self.criteria]
        if len(ids) != len(set(ids)):
            raise ValueError("criterion ids must be unique")
        return self

    def criterion(self, criterion_id: str) -> Criterion:
        """Return a criterion by identifier."""

        for criterion in self.criteria:
            if criterion.id == criterion_id:
                return criterion
        raise KeyError(criterion_id)

    @property
    def fingerprint(self) -> str:
        """Return a stable hash of the rubric content, excluding identity."""

        content = self.model_dump(
            mode="json",
            exclude={"id", "version"},
        )
        encoded = json.dumps(
            content,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True, init=False)
class Score:
    """A scalar grading result with recursively composable components."""

    value: float
    components: Mapping[str, Score]
    metadata: Mapping[str, Any]
    feedback: str | None

    def __init__(
        self,
        value: float,
        components: Mapping[str, float | Score] | None = None,
        metadata: Mapping[str, Any] | None = None,
        *,
        feedback: str | None = None,
    ) -> None:
        resolved_value = float(value)
        if not math.isfinite(resolved_value):
            raise ValueError(f"score must be finite, got {value!r}")
        resolved_components = {
            name: Score.from_value(component)
            for name, component in (components or {}).items()
        }
        if any(not name for name in resolved_components):
            raise ValueError("score component names must be non-empty")
        if feedback is not None:
            feedback = feedback.strip()
            if not feedback:
                raise ValueError("score feedback must not be empty")
        object.__setattr__(self, "value", resolved_value)
        object.__setattr__(self, "components", resolved_components)
        object.__setattr__(self, "metadata", dict(metadata or {}))
        object.__setattr__(self, "feedback", feedback)

    @classmethod
    def from_value(cls, value: float | Score) -> Score:
        """Normalize a scalar or existing score."""

        return value if isinstance(value, Score) else cls(float(value))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Score:
        """Deserialize a score from its dictionary representation."""

        if "value" not in value:
            raise ValueError("serialized score must contain 'value'")
        raw_components = value.get("components", {})
        raw_metadata = value.get("metadata", {})
        if not isinstance(raw_components, Mapping):
            raise TypeError("serialized score components must be a mapping")
        if not isinstance(raw_metadata, Mapping):
            raise TypeError("serialized score metadata must be a mapping")
        components: dict[str, Score] = {}
        for name, component in raw_components.items():
            if not isinstance(name, str):
                raise TypeError("serialized score component names must be strings")
            if isinstance(component, Score):
                components[name] = component
            elif isinstance(component, Mapping):
                components[name] = cls.from_dict(component)
            else:
                components[name] = cls.from_value(cast(Any, component))
        feedback = value.get("feedback")
        if feedback is not None and not isinstance(feedback, str):
            raise TypeError("serialized score feedback must be a string or None")
        return cls(
            cast(Any, value["value"]),
            components,
            cast(Mapping[str, Any], raw_metadata),
            feedback=feedback,
        )

    @overload
    @classmethod
    def from_info(cls, info: Mapping[str, Any], *, default: Score) -> Score: ...

    @overload
    @classmethod
    def from_info(
        cls,
        info: Mapping[str, Any],
        *,
        default: None = None,
    ) -> Score | None: ...

    @classmethod
    def from_info(
        cls,
        info: Mapping[str, Any],
        *,
        default: Score | None = None,
    ) -> Score | None:
        """Read a structured score from an environment info mapping."""

        value = info.get("score")
        if value is None:
            return default
        if isinstance(value, Score):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("info['score'] must be a Score or mapping")
        return cls.from_dict(value)

    @property
    def component_values(self) -> dict[str, float]:
        """Return the immediate component values."""

        return {name: component.value for name, component in self.components.items()}

    def to_dict(self) -> dict[str, Any]:
        """Return a recursively serializable dictionary."""

        result: dict[str, Any] = {
            "value": self.value,
            "components": {
                name: component.to_dict() for name, component in self.components.items()
            },
            "metadata": dict(self.metadata),
        }
        if self.feedback is not None:
            result["feedback"] = self.feedback
        return result

    def as_info(self) -> dict[str, Any]:
        """Return an environment ``info`` payload containing this score."""

        return {"score": self.to_dict()}


ScoreValue: TypeAlias = float | Score


def _with_rubric_metadata(score: Score, rubric: Rubric | None) -> Score:
    if rubric is None:
        return score
    metadata = dict(score.metadata)
    metadata.setdefault("rubric_fingerprint", rubric.fingerprint)
    if rubric.id is not None:
        metadata.setdefault("rubric_id", rubric.id)
    if rubric.version is not None:
        metadata.setdefault("rubric_version", rubric.version)
    if metadata == score.metadata:
        return score
    return Score(
        score.value,
        score.components,
        metadata,
        feedback=score.feedback,
    )


def _sync_score(value: object, *, source: str) -> Score:
    if inspect.isawaitable(value):
        close = getattr(value, "close", None)
        if callable(close):
            close()
        raise TypeError(f"{source} returned an awaitable; use an async grader")
    return Score.from_value(cast(Any, value))


class Grader(Generic[InputT], ABC):
    """Synchronous grading contract.

    The input type is application-defined. It may be a response, trajectory,
    tool trace, comparison, or richer object containing reference material.
    ``input_space`` declares and validates that type at runtime.
    """

    input_space: Space[InputT]

    @final
    def grade(
        self,
        input: InputT,
        *,
        rubric: Rubric | None = None,
    ) -> Score:
        """Validate and grade one input, optionally according to a rubric."""

        self._validate_input(input)
        return self._grade(input, rubric=rubric)

    @abstractmethod
    def _grade(
        self,
        input: InputT,
        *,
        rubric: Rubric | None = None,
    ) -> Score:
        raise NotImplementedError

    def _validate_input(self, input: InputT) -> None:
        try:
            valid = self.input_space.contains(input)
        except AttributeError as error:
            raise TypeError("grader must define an input_space") from error
        if not valid:
            raise ValueError(
                f"grader input is outside input_space: {self.input_space!r}"
            )

    def bind(self, rubric: Rubric) -> Grader[InputT]:
        """Return a grader with ``rubric`` fixed for later calls."""

        return _BoundGrader(self, rubric)


class AsyncGrader(Generic[InputT], ABC):
    """Asynchronous grading contract with the same value-level semantics."""

    input_space: Space[InputT]

    @final
    async def grade(
        self,
        input: InputT,
        *,
        rubric: Rubric | None = None,
    ) -> Score:
        """Validate and asynchronously grade one input."""

        self._validate_input(input)
        return await self._grade(input, rubric=rubric)

    @abstractmethod
    async def _grade(
        self,
        input: InputT,
        *,
        rubric: Rubric | None = None,
    ) -> Score:
        raise NotImplementedError

    def _validate_input(self, input: InputT) -> None:
        try:
            valid = self.input_space.contains(input)
        except AttributeError as error:
            raise TypeError("grader must define an input_space") from error
        if not valid:
            raise ValueError(
                f"grader input is outside input_space: {self.input_space!r}"
            )

    def bind(self, rubric: Rubric) -> AsyncGrader[InputT]:
        """Return an async grader with ``rubric`` fixed for later calls."""

        return _BoundAsyncGrader(self, rubric)


class _BoundGrader(Grader[InputT]):
    def __init__(self, grader: Grader[InputT], rubric: Rubric) -> None:
        self.grader = grader
        self.rubric = rubric
        self.input_space = grader.input_space

    def _grade(
        self,
        input: InputT,
        *,
        rubric: Rubric | None = None,
    ) -> Score:
        if rubric is not None and rubric != self.rubric:
            raise ValueError("a bound grader cannot be used with another rubric")
        return self.grader.grade(input, rubric=self.rubric)


class _BoundAsyncGrader(AsyncGrader[InputT]):
    def __init__(self, grader: AsyncGrader[InputT], rubric: Rubric) -> None:
        self.grader = grader
        self.rubric = rubric
        self.input_space = grader.input_space

    async def _grade(
        self,
        input: InputT,
        *,
        rubric: Rubric | None = None,
    ) -> Score:
        if rubric is not None and rubric != self.rubric:
            raise ValueError("a bound grader cannot be used with another rubric")
        return await self.grader.grade(input, rubric=self.rubric)


GradeCallable: TypeAlias = Callable[
    [CallableInputT, Rubric | None],
    ScoreValue,
]
AsyncGradeCallable: TypeAlias = Callable[
    [CallableInputT, Rubric | None],
    ScoreValue | Awaitable[ScoreValue],
]


class CallableGrader(Grader[CallableInputT]):
    """Adapt a synchronous callable to the grader contract."""

    def __init__(
        self,
        function: GradeCallable[CallableInputT],
        *,
        input_space: Space[CallableInputT],
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.function = function
        self.input_space = input_space
        self.metadata = dict(metadata or {})

    def _grade(
        self,
        input: CallableInputT,
        *,
        rubric: Rubric | None = None,
    ) -> Score:
        score = _sync_score(
            self.function(input, rubric),
            source="grade callable",
        )
        metadata = dict(self.metadata)
        metadata.update(score.metadata)
        score = Score(
            score.value,
            score.components,
            metadata,
            feedback=score.feedback,
        )
        return _with_rubric_metadata(score, rubric)


class AsyncCallableGrader(AsyncGrader[CallableInputT]):
    """Adapt a synchronous or asynchronous callable to the async contract."""

    def __init__(
        self,
        function: AsyncGradeCallable[CallableInputT],
        *,
        input_space: Space[CallableInputT],
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.function = function
        self.input_space = input_space
        self.metadata = dict(metadata or {})

    async def _grade(
        self,
        input: CallableInputT,
        *,
        rubric: Rubric | None = None,
    ) -> Score:
        value = self.function(input, rubric)
        if inspect.isawaitable(value):
            value = await value
        score = Score.from_value(value)
        metadata = dict(self.metadata)
        metadata.update(score.metadata)
        score = Score(
            score.value,
            score.components,
            metadata,
            feedback=score.feedback,
        )
        return _with_rubric_metadata(score, rubric)


Aggregator: TypeAlias = Callable[[Rubric, Mapping[str, Score]], float]
CriterionScorer: TypeAlias = Callable[
    [RubricInputT, Criterion],
    ScoreValue,
]
AsyncCriterionScorer: TypeAlias = Callable[
    [RubricInputT, Criterion],
    ScoreValue | Awaitable[ScoreValue],
]


def weighted_sum(rubric: Rubric, components: Mapping[str, Score]) -> float:
    """Return the weighted sum of criterion component scores."""

    return sum(
        criterion.weight * components[criterion.id].value
        for criterion in rubric.criteria
    )


def weighted_mean(rubric: Rubric, components: Mapping[str, Score]) -> float:
    """Return the normalized weighted mean of criterion scores."""

    total_weight = sum(criterion.weight for criterion in rubric.criteria)
    return weighted_sum(rubric, components) / total_weight


def all_pass(rubric: Rubric, components: Mapping[str, Score]) -> float:
    """Return one only when every criterion score is at least one."""

    return float(all(components[item.id].value >= 1.0 for item in rubric.criteria))


def asymmetric_mean(
    rubric: Rubric,
    components: Mapping[str, Score],
    *,
    bonus_weight: float = 1.0,
    penalty_weight: float = 1.0,
) -> float:
    """Aggregate ``required``, ``bonus``, and ``penalty`` categories.

    Penalty criteria should describe the desired safe behavior: their degree of
    failure is subtracted. Categories are optional rubric vocabulary rather
    than a restriction on all criteria.
    """

    if not math.isfinite(bonus_weight) or bonus_weight < 0:
        raise ValueError("bonus_weight must be finite and non-negative")
    if not math.isfinite(penalty_weight) or penalty_weight < 0:
        raise ValueError("penalty_weight must be finite and non-negative")

    categories = {criterion.category or "required" for criterion in rubric.criteria}
    unknown = sorted(categories - {"required", "bonus", "penalty"})
    if unknown:
        raise ValueError(f"unsupported criterion categories: {', '.join(unknown)}")

    def mean(category: str, *, failure: bool = False) -> float:
        criteria = [
            item
            for item in rubric.criteria
            if (item.category or "required") == category
        ]
        total = sum(item.weight for item in criteria)
        if total == 0:
            return 0.0
        return (
            sum(
                item.weight
                * (
                    1.0 - components[item.id].value
                    if failure
                    else components[item.id].value
                )
                for item in criteria
            )
            / total
        )

    return (
        mean("required")
        + bonus_weight * mean("bonus")
        - penalty_weight * mean("penalty", failure=True)
    )


class RubricGrader(Grader[RubricInputT]):
    """Grade each rubric criterion independently and aggregate the results."""

    def __init__(
        self,
        scorer: CriterionScorer[RubricInputT]
        | Mapping[str, CriterionScorer[RubricInputT]],
        *,
        input_space: Space[RubricInputT],
        overrides: Mapping[str, CriterionScorer[RubricInputT]] | None = None,
        aggregate: Aggregator = weighted_mean,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if isinstance(scorer, Mapping):
            configured = dict(scorer)
            configured.update(overrides or {})
            if not configured:
                raise ValueError("a rubric grader requires at least one scorer")
            self._scorer: CriterionScorer[RubricInputT] | None = None
            self._overrides = configured
        else:
            self._scorer = scorer
            self._overrides = dict(overrides or {})
        self.input_space = input_space
        self._aggregate = aggregate
        self._metadata = dict(metadata or {})

    def _grade(
        self,
        input: RubricInputT,
        *,
        rubric: Rubric | None = None,
    ) -> Score:
        resolved_rubric = self._require_rubric(rubric)
        self._validate_rubric(resolved_rubric)
        components: dict[str, Score] = {}
        for criterion in resolved_rubric.criteria:
            components[criterion.id] = _sync_score(
                self._scorer_for(criterion.id)(input, criterion),
                source=f"criterion scorer {criterion.id!r}",
            )
        return self._make_score(resolved_rubric, components)

    def _require_rubric(self, rubric: Rubric | None) -> Rubric:
        if rubric is None:
            raise ValueError("RubricGrader requires a rubric")
        return rubric

    def _scorer_for(self, criterion_id: str) -> CriterionScorer[RubricInputT]:
        scorer = self._overrides.get(criterion_id, self._scorer)
        if scorer is None:
            raise ValueError(f"no scorer configured for criterion: {criterion_id}")
        return scorer

    def _validate_rubric(self, rubric: Rubric) -> None:
        if self._scorer is not None:
            return
        missing = [
            criterion.id
            for criterion in rubric.criteria
            if criterion.id not in self._overrides
        ]
        if missing:
            raise ValueError(f"no scorer configured for criteria: {', '.join(missing)}")

    def _make_score(
        self,
        rubric: Rubric,
        components: dict[str, Score],
    ) -> Score:
        return _with_rubric_metadata(
            Score(
                self._aggregate(rubric, components),
                components,
                self._metadata,
            ),
            rubric,
        )


class AsyncRubricGrader(AsyncGrader[RubricInputT]):
    """Asynchronously grade rubric criteria with bounded implementation freedom."""

    def __init__(
        self,
        scorer: AsyncCriterionScorer[RubricInputT]
        | Mapping[str, AsyncCriterionScorer[RubricInputT]],
        *,
        input_space: Space[RubricInputT],
        overrides: Mapping[str, AsyncCriterionScorer[RubricInputT]] | None = None,
        aggregate: Aggregator = weighted_mean,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if isinstance(scorer, Mapping):
            configured = dict(scorer)
            configured.update(overrides or {})
            if not configured:
                raise ValueError("an async rubric grader requires at least one scorer")
            self._scorer: AsyncCriterionScorer[RubricInputT] | None = None
            self._overrides = configured
        else:
            self._scorer = scorer
            self._overrides = dict(overrides or {})
        self.input_space = input_space
        self._aggregate = aggregate
        self._metadata = dict(metadata or {})

    async def _grade(
        self,
        input: RubricInputT,
        *,
        rubric: Rubric | None = None,
    ) -> Score:
        if rubric is None:
            raise ValueError("AsyncRubricGrader requires a rubric")
        self._validate_rubric(rubric)

        async def resolve(criterion: Criterion) -> tuple[str, Score]:
            value = self._scorer_for(criterion.id)(input, criterion)
            if inspect.isawaitable(value):
                value = await value
            return criterion.id, Score.from_value(value)

        resolved = await asyncio.gather(*(resolve(item) for item in rubric.criteria))
        components = dict(resolved)
        return _with_rubric_metadata(
            Score(
                self._aggregate(rubric, components),
                components,
                self._metadata,
            ),
            rubric,
        )

    def _scorer_for(self, criterion_id: str) -> AsyncCriterionScorer[RubricInputT]:
        scorer = self._overrides.get(criterion_id, self._scorer)
        if scorer is None:
            raise ValueError(f"no scorer configured for criterion: {criterion_id}")
        return scorer

    def _validate_rubric(self, rubric: Rubric) -> None:
        if self._scorer is not None:
            return
        missing = [
            criterion.id
            for criterion in rubric.criteria
            if criterion.id not in self._overrides
        ]
        if missing:
            raise ValueError(f"no scorer configured for criteria: {', '.join(missing)}")


__all__ = [
    "Aggregator",
    "AsyncCallableGrader",
    "AsyncGrader",
    "AsyncRubricGrader",
    "Criterion",
    "CriterionScorer",
    "CallableGrader",
    "Grader",
    "Level",
    "Rubric",
    "RubricGrader",
    "Score",
    "ScoreValue",
    "all_pass",
    "asymmetric_mean",
    "weighted_mean",
    "weighted_sum",
]
