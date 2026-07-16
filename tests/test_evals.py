from __future__ import annotations

from typing import Any, cast

import pytest

from rolloutlib.evals import Benchmark, Evaluation, run_benchmark, run_benchmarks
from rolloutlib.graders import Score


class ToyEnv:
    def __init__(self, value: int) -> None:
        self.value = value
        self.closed = False

    def close(self) -> None:
        self.closed = True


def test_sync_benchmark_aggregates_scores_components_errors_and_truncation() -> None:
    def evaluate(environment: ToyEnv) -> Evaluation:
        if environment.value == 2:
            raise RuntimeError("broken example")
        if environment.value == 3:
            return Evaluation(
                score=Score(0.5, {"correct": 1.0}),
                truncated=True,
            )
        item = environment.value
        return Evaluation(Score(float(item), {"correct": float(item)}))

    result = run_benchmark(
        Benchmark("toy", [0, 1, 2, 3], ToyEnv),
        evaluate,
    )

    assert result.name == "toy"
    assert result.score == pytest.approx(0.375)
    assert result.score_completed == pytest.approx(0.5)
    assert result.components == {"correct": pytest.approx(0.5)}
    assert result.num_examples == 4
    assert result.num_scored == 3
    assert result.num_completed == 2
    assert result.num_errors == 1
    assert result.num_truncated == 1
    assert result.records[2].error == "broken example"


def test_sync_benchmark_rejects_async_callback() -> None:
    async def evaluate(_: ToyEnv) -> Evaluation:
        return Evaluation(Score(1.0, {}))

    with pytest.raises(TypeError, match="must be synchronous"):
        run_benchmark(Benchmark("toy", [1], ToyEnv), cast(Any, evaluate))


def test_sync_benchmark_requires_evaluation_result() -> None:
    with pytest.raises(TypeError, match="must return Evaluation"):
        run_benchmark(
            Benchmark("toy", [1], ToyEnv),
            cast(Any, lambda _: Score(1.0, {})),
            fail_fast=True,
        )


def test_run_benchmarks_runs_named_benchmarks_sequentially() -> None:
    results = run_benchmarks(
        [
            Benchmark("a", [1], ToyEnv),
            Benchmark("b", [2, 3], ToyEnv),
        ],
        evaluate=lambda environment: Evaluation(
            Score(float(environment.value == 1), {})
        ),
    )

    assert set(results) == {"a", "b"}
    assert results["a"].score == 1.0
    assert results["b"].score == 0.0
