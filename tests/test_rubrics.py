from __future__ import annotations

import asyncio

import pytest
from pydantic import ValidationError

from rolloutlib.graders import CompositeGrader, Criterion, Rubric


def test_rubric_is_declarative_and_serializable() -> None:
    rubric = Rubric(
        criteria=(
            Criterion(
                id="correctness",
                description="The response is correct.",
                references=("Reference answer",),
            ),
        ),
        id="answer-rubric",
        instructions="Use the reference answer.",
        metadata={"version": "1"},
    )

    assert rubric.criterion("correctness").weight == 1.0
    assert len(rubric.fingerprint) == 64
    assert rubric.model_dump(mode="json") == {
        "id": "answer-rubric",
        "criteria": [
            {
                "id": "correctness",
                "description": "The response is correct.",
                "weight": 1.0,
                "kind": "required",
                "references": ["Reference answer"],
                "metadata": {},
            }
        ],
        "instructions": "Use the reference answer.",
        "metadata": {"version": "1"},
    }


def test_rubric_rejects_empty_or_duplicate_criteria() -> None:
    with pytest.raises(ValidationError, match="at least 1"):
        Rubric(criteria=())

    with pytest.raises(ValidationError, match="unique"):
        Rubric(
            criteria=(
                Criterion(id="same", description="First."),
                Criterion(id="same", description="Second."),
            )
        )


def test_criterion_rejects_invalid_weight_and_text() -> None:
    with pytest.raises(ValidationError):
        Criterion(id="", description="Valid.")
    with pytest.raises(ValidationError):
        Criterion(id="valid", description="Valid.", weight=-1)
    with pytest.raises(ValidationError, match="finite"):
        Criterion(id="valid", description="Valid.", weight=float("inf"))


def test_async_criterion_graders_run_concurrently() -> None:
    async def run() -> None:
        release = asyncio.Event()
        both_started = asyncio.Event()
        started = 0

        async def criterion_grader(value: object, criterion: Criterion) -> float:
            nonlocal started
            started += 1
            if started == 2:
                both_started.set()
            await release.wait()
            return 1.0

        rubric = Rubric(
            criteria=(
                Criterion(id="first", description="First."),
                Criterion(id="second", description="Second."),
            )
        )
        grader = CompositeGrader(
            {"first": criterion_grader, "second": criterion_grader}
        )
        evaluation = asyncio.create_task(grader.ascore(object(), rubric))

        await asyncio.wait_for(both_started.wait(), timeout=1)
        assert not evaluation.done()
        release.set()
        score = await evaluation
        assert score.value == 1.0
        assert score.component_values == {"first": 1.0, "second": 1.0}

    asyncio.run(run())
