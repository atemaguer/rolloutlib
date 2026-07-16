"""Evaluation primitives for datasets, benchmarks, and user-owned runners.

The evaluation layer deliberately does not sample a policy or know about a
model SDK. A :class:`Benchmark` supplies an item collection and an environment
factory. The user-owned callback is an escape hatch for policy interaction on
a fresh environment and returns an :class:`Evaluation`.
"""

from __future__ import annotations

import inspect
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import islice
from typing import Any, Generic, Protocol, TypeVar

from ..graders import Score


ItemT = TypeVar("ItemT")
EnvironmentT = TypeVar("EnvironmentT", contravariant=True)


@dataclass(frozen=True, slots=True)
class Evaluation:
    """Per-example evaluation output.

    ``truncated`` distinguishes an incomplete run from a completed low score.
    ``metadata`` can carry task identifiers, artifact references, or other
    diagnostics without imposing a benchmark-specific schema.
    """

    score: Score
    truncated: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)


class EvaluationCallback(Protocol[EnvironmentT]):
    """User-owned evaluation of one fresh environment instance."""

    def __call__(self, environment: EnvironmentT, /) -> Evaluation: ...


@dataclass(frozen=True, slots=True)
class EvaluationRecord:
    """One benchmark example's successful or failed evaluation."""

    index: int
    item_id: str | None = None
    score: Score | None = None
    truncated: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass(frozen=True, slots=True)
class Benchmark(Generic[ItemT]):
    """A named item collection and a fresh-environment factory."""

    name: str
    items: Sequence[ItemT]
    make_env: Callable[[ItemT], Any]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    item_id: Callable[[ItemT], str] | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("benchmark name must be non-empty")
        object.__setattr__(self, "items", tuple(self.items))
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """Aggregated results for one benchmark run."""

    name: str
    score: float
    score_completed: float | None
    components: Mapping[str, float]
    num_examples: int
    num_scored: int
    num_completed: int
    num_errors: int
    num_truncated: int
    elapsed_seconds: float
    records: tuple[EvaluationRecord, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)


def _limit_items(items: Iterable[ItemT], limit: int | None) -> list[ItemT]:
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")
    if limit is None:
        return list(items)
    return list(islice(items, limit))


def _as_evaluation(value: object) -> Evaluation:
    if isinstance(value, Evaluation):
        return value
    raise TypeError("evaluation callback must return Evaluation")


def _is_async_callback(callback: object) -> bool:
    return inspect.iscoroutinefunction(callback) or inspect.iscoroutinefunction(
        getattr(callback, "__call__", None)
    )


def _close_awaitable(value: object) -> None:
    close = getattr(value, "close", None)
    if callable(close):
        close()


def _close_environment(environment: object) -> None:
    close = getattr(environment, "close", None)
    if not callable(close):
        return
    result = close()
    if inspect.isawaitable(result):
        _close_awaitable(result)
        raise TypeError("synchronous benchmark environments must have sync close()")


def _aggregate(
    benchmark: Benchmark[Any],
    records: list[EvaluationRecord],
    elapsed_seconds: float,
) -> BenchmarkResult:
    num_examples = len(records)
    successful = [record for record in records if record.score is not None]
    completed = [record for record in successful if not record.truncated]
    errors = [record for record in records if record.error is not None]
    truncated = [record for record in records if record.truncated]

    total = sum(record.score.value for record in successful if record.score is not None)
    completed_total = sum(
        record.score.value for record in completed if record.score is not None
    )
    component_totals: dict[str, float] = {}
    for record in successful:
        if record.score is None:
            continue
        for name, value in record.score.components.items():
            component_totals[name] = component_totals.get(name, 0.0) + value.value

    denominator = num_examples or 1
    completed_score = completed_total / len(completed) if completed else None
    return BenchmarkResult(
        name=benchmark.name,
        score=total / denominator,
        score_completed=completed_score,
        components={
            name: value / denominator for name, value in component_totals.items()
        },
        num_examples=num_examples,
        num_scored=len(successful),
        num_completed=len(completed),
        num_errors=len(errors),
        num_truncated=len(truncated),
        elapsed_seconds=elapsed_seconds,
        records=tuple(records),
        metadata=dict(benchmark.metadata),
    )


def run_benchmark(
    benchmark: Benchmark[ItemT],
    evaluate: EvaluationCallback[Any],
    *,
    limit: int | None = None,
    fail_fast: bool = False,
) -> BenchmarkResult:
    """Run a benchmark with a synchronous user-owned evaluation callback.

    ``evaluate`` receives a fresh environment for each example and owns the
    policy/environment interaction. Rolloutlib never samples a model itself.
    """

    if _is_async_callback(evaluate):
        raise TypeError("benchmark callback must be synchronous")
    items = _limit_items(benchmark.items, limit)
    started = time.perf_counter()
    records: list[EvaluationRecord] = []
    for index, item in enumerate(items):
        item_id = benchmark.item_id(item) if benchmark.item_id else str(index)
        try:
            environment = benchmark.make_env(item)
            try:
                value = evaluate(environment)
            finally:
                _close_environment(environment)
            if inspect.isawaitable(value):
                _close_awaitable(value)
                raise TypeError("benchmark callback must be synchronous")
            evaluation = _as_evaluation(value)
            records.append(
                EvaluationRecord(
                    index=index,
                    item_id=item_id,
                    score=evaluation.score,
                    truncated=evaluation.truncated,
                    metadata=evaluation.metadata,
                )
            )
        except Exception as exc:
            if fail_fast:
                raise
            records.append(
                EvaluationRecord(
                    index=index,
                    item_id=item_id,
                    metadata={"error_type": type(exc).__name__},
                    error=str(exc),
                )
            )
    return _aggregate(benchmark, records, time.perf_counter() - started)


def run_benchmarks(
    benchmarks: Iterable[Benchmark[Any]],
    evaluate: EvaluationCallback[Any],
    *,
    limit: int | None = None,
    fail_fast: bool = False,
) -> dict[str, BenchmarkResult]:
    """Run multiple synchronous benchmarks sequentially by name."""

    results: dict[str, BenchmarkResult] = {}
    for benchmark in benchmarks:
        if benchmark.name in results:
            raise ValueError(f"duplicate benchmark name: {benchmark.name}")
        results[benchmark.name] = run_benchmark(
            benchmark,
            evaluate,
            limit=limit,
            fail_fast=fail_fast,
        )
    return results


__all__ = [
    "Benchmark",
    "BenchmarkResult",
    "EvaluationCallback",
    "Evaluation",
    "EvaluationRecord",
    "run_benchmark",
    "run_benchmarks",
]
