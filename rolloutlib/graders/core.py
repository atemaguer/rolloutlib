"""Rubric specifications, graders, and score results.

A rubric describes what should be evaluated. A grader applies that rubric to
an application-defined input and returns a score. Environments use the scalar
score value as reward and preserve the complete score in step ``info``.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import math
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any, Generic, Protocol, Self, TypeAlias, TypeVar, cast, overload

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


InputT = TypeVar("InputT", contravariant=True)
RubricT = TypeVar("RubricT", contravariant=True)
CompositeInputT = TypeVar("CompositeInputT")


class Criterion(BaseModel):
    """One independently identifiable requirement in a rubric."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    description: str
    weight: float = Field(default=1.0, ge=0.0)
    kind: str = "required"
    references: tuple[str, ...] = ()
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("id", "description", "kind")
    @classmethod
    def _non_empty_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be empty")
        return value

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


class Rubric(BaseModel):
    """A declarative collection of criteria and grading instructions."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str | None = None
    criteria: tuple[Criterion, ...] = Field(min_length=1)
    instructions: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("id", "instructions")
    @classmethod
    def _non_empty_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("must not be empty")
        return value

    @model_validator(mode="after")
    def _unique_criterion_ids(self) -> Self:
        ids = [criterion.id for criterion in self.criteria]
        if len(ids) != len(set(ids)):
            raise ValueError("criterion ids must be unique")
        return self

    def criterion(self, criterion_id: str) -> Criterion:
        """Return a criterion by id."""

        for criterion in self.criteria:
            if criterion.id == criterion_id:
                return criterion
        raise KeyError(criterion_id)

    @property
    def fingerprint(self) -> str:
        """Return a deterministic fingerprint of the rubric's content."""

        content = self.model_dump(mode="python", exclude={"id"})
        encoded = json.dumps(
            content,
            sort_keys=True,
            separators=(",", ":"),
            default=repr,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True, init=False)
class Score:
    """A scalar score with recursively composable component scores."""

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
        """Normalize a scalar or existing score into a :class:`Score`."""

        return value if isinstance(value, Score) else cls(float(value))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Score:
        """Deserialize a score from its structured dictionary form."""

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
    def from_info(
        cls,
        info: Mapping[str, Any],
        *,
        default: Score,
    ) -> Score: ...

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
        """Read an environment-produced score from Gymnasium ``info``."""

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
        """Return the immediate component names and scalar values."""

        return {name: component.value for name, component in self.components.items()}

    def to_dict(self) -> dict[str, Any]:
        """Return a structured, serialization-friendly representation."""

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
        """Return an ``info`` payload suitable for an environment step."""

        return {"score": self.to_dict()}


ScoreValue: TypeAlias = float | Score


class Grader(Protocol[InputT, RubricT]):
    """Callable that applies a rubric to an application-defined input."""

    def __call__(
        self,
        input: InputT,
        rubric: RubricT,
        /,
    ) -> ScoreValue | Awaitable[ScoreValue]: ...


Aggregator: TypeAlias = Callable[[Rubric, Mapping[str, Score]], float]


def weighted_sum(rubric: Rubric, components: Mapping[str, Score]) -> float:
    """Return the weighted sum of criterion component scores."""

    return sum(
        criterion.weight * components[criterion.id].value
        for criterion in rubric.criteria
    )


def weighted_mean(rubric: Rubric, components: Mapping[str, Score]) -> float:
    """Return the normalized weighted mean of criterion component scores."""

    total_weight = sum(criterion.weight for criterion in rubric.criteria)
    if total_weight == 0:
        raise ValueError("weighted mean requires at least one positive weight")
    return weighted_sum(rubric, components) / total_weight


def all_pass(rubric: Rubric, components: Mapping[str, Score]) -> float:
    """Return one only when every criterion has a value of at least one."""

    return float(all(components[item.id].value >= 1.0 for item in rubric.criteria))


def asymmetric_mean(
    rubric: Rubric,
    components: Mapping[str, Score],
    *,
    bonus_weight: float = 1.0,
    penalty_weight: float = 1.0,
) -> float:
    """Aggregate required, bonus, and penalty criteria asymmetrically.

    Required and bonus criteria contribute their degree of satisfaction.
    Penalty criteria subtract their degree of failure. Criterion weights apply
    within each kind; ``bonus_weight`` and ``penalty_weight`` scale those terms.
    """

    if not math.isfinite(bonus_weight) or bonus_weight < 0:
        raise ValueError("bonus_weight must be finite and non-negative")
    if not math.isfinite(penalty_weight) or penalty_weight < 0:
        raise ValueError("penalty_weight must be finite and non-negative")

    supported = {"required", "bonus", "penalty"}
    unknown = sorted({item.kind for item in rubric.criteria} - supported)
    if unknown:
        raise ValueError(f"unsupported criterion kinds: {', '.join(unknown)}")

    def mean(kind: str, *, failure: bool = False) -> float:
        criteria = [item for item in rubric.criteria if item.kind == kind]
        total = sum(item.weight for item in criteria)
        if total == 0:
            return 0.0
        return sum(
            item.weight
            * (
                1.0 - components[item.id].value
                if failure
                else components[item.id].value
            )
            for item in criteria
        ) / total

    return (
        mean("required")
        + bonus_weight * mean("bonus")
        - penalty_weight * mean("penalty", failure=True)
    )


class CompositeGrader(Generic[CompositeInputT]):
    """Apply criterion graders and aggregate their component scores.

    One default grader can evaluate every criterion, while ``overrides`` can
    provide specialized deterministic or model-based graders by criterion id.
    Passing a mapping as the first argument retains the explicit per-id form.
    """

    def __init__(
        self,
        grader: Grader[CompositeInputT, Criterion]
        | Mapping[str, Grader[CompositeInputT, Criterion]],
        *,
        overrides: Mapping[str, Grader[CompositeInputT, Criterion]] | None = None,
        aggregate: Aggregator = weighted_mean,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if isinstance(grader, Mapping):
            configured = dict(grader)
            configured.update(overrides or {})
            if not configured:
                raise ValueError("a composite grader requires at least one grader")
            self._grader: Grader[CompositeInputT, Criterion] | None = None
            self._overrides = configured
        else:
            if not callable(grader):
                raise TypeError("grader must be callable or a mapping of graders")
            self._grader = grader
            self._overrides = dict(overrides or {})
        if any(not name for name in self._overrides):
            raise ValueError("grader names must be non-empty")
        self._aggregate = aggregate
        self._metadata = dict(metadata or {})

    @property
    def names(self) -> tuple[str, ...]:
        """Return criterion ids with explicitly configured overrides."""

        return tuple(self._overrides)

    def __call__(self, input: CompositeInputT, rubric: Rubric, /) -> Score:
        return self.score(input, rubric)

    def score(self, input: CompositeInputT, rubric: Rubric, /) -> Score:
        """Apply synchronous criterion graders."""

        self._validate_rubric(rubric)
        components: dict[str, Score] = {}
        for criterion in rubric.criteria:
            value = self._grader_for(criterion.id)(input, criterion)
            if inspect.isawaitable(value):
                close = getattr(value, "close", None)
                if callable(close):
                    close()
                raise TypeError(
                    f"grader {criterion.id!r} returned an awaitable; use ascore()"
                )
            components[criterion.id] = self._coerce_score(criterion.id, value)
        return self._make_score(rubric, components)

    async def ascore(self, input: CompositeInputT, rubric: Rubric, /) -> Score:
        """Apply synchronous or asynchronous criterion graders concurrently."""

        self._validate_rubric(rubric)

        async def resolve(criterion: Criterion) -> tuple[str, Score]:
            value = self._grader_for(criterion.id)(input, criterion)
            if inspect.isawaitable(value):
                value = await value
            return criterion.id, self._coerce_score(criterion.id, value)

        resolved = await asyncio.gather(*(resolve(item) for item in rubric.criteria))
        return self._make_score(rubric, dict(resolved))

    def _grader_for(self, criterion_id: str) -> Grader[CompositeInputT, Criterion]:
        grader = self._overrides.get(criterion_id, self._grader)
        if grader is None:
            raise ValueError(f"no grader configured for rubric criterion: {criterion_id}")
        return grader

    def _validate_rubric(self, rubric: Rubric) -> None:
        if self._grader is not None:
            return
        missing = [
            criterion.id
            for criterion in rubric.criteria
            if criterion.id not in self._overrides
        ]
        if missing:
            raise ValueError(
                f"no grader configured for rubric criteria: {', '.join(missing)}"
            )

    def _make_score(
        self,
        rubric: Rubric,
        components: dict[str, Score],
    ) -> Score:
        metadata = dict(self._metadata)
        metadata.setdefault("rubric_fingerprint", rubric.fingerprint)
        if rubric.id is not None:
            metadata.setdefault("rubric_id", rubric.id)
        return Score(
            value=self._aggregate(rubric, components),
            components=components,
            metadata=metadata,
        )

    @staticmethod
    def _coerce_score(name: str, value: object) -> Score:
        try:
            return Score.from_value(cast(Any, value))
        except (TypeError, ValueError) as exc:
            raise TypeError(f"grader {name!r} did not return a numeric score") from exc


__all__ = [
    "Aggregator",
    "CompositeGrader",
    "Criterion",
    "Grader",
    "Rubric",
    "Score",
    "ScoreValue",
    "all_pass",
    "asymmetric_mean",
    "weighted_mean",
    "weighted_sum",
]
