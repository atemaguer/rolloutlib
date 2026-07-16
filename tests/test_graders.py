from __future__ import annotations

import asyncio

import pytest

from rolloutlib.graders import (
    CompositeGrader,
    Criterion,
    Grader,
    Rubric,
    Score,
    all_pass,
    asymmetric_mean,
    weighted_sum,
)


def answer_rubric() -> Rubric:
    return Rubric(
        criteria=(
            Criterion(
                id="correctness",
                description="The answer is correct.",
                weight=2.0,
            ),
            Criterion(
                id="format",
                description="The answer uses the requested format.",
                weight=0.25,
            ),
        ),
        instructions="Grade only the submitted answer.",
        metadata={"source": "test"},
    )


def test_composite_grader_applies_a_declarative_rubric() -> None:
    grader = CompositeGrader[str](
        lambda answer, criterion: 0.5 if answer.isdigit() else 0.0,
        overrides={
            "correctness": lambda answer, criterion: Score(
                float(answer == "42"),
                feedback="The answer is correct.",
            ),
        },
        metadata={"grader": "local"},
    )

    score = grader.score("42", answer_rubric())

    assert score.value == pytest.approx(2.125 / 2.25)
    assert score.component_values == {"correctness": 1.0, "format": 0.5}
    assert score.components["correctness"].feedback == "The answer is correct."
    assert score.metadata["grader"] == "local"
    restored = Score.from_info(score.as_info())
    assert restored == score


def test_composite_grader_supports_custom_aggregation() -> None:
    grader = CompositeGrader[str](
        {
            "correctness": lambda answer, criterion: float(answer == "42"),
            "format": lambda answer, criterion: 0.5,
        },
        aggregate=weighted_sum,
    )

    assert grader("42", answer_rubric()).value == pytest.approx(2.125)


def test_sync_grading_rejects_async_criterion_graders() -> None:
    async def correctness(answer: str, criterion: Criterion) -> float:
        return 1.0

    grader = CompositeGrader({"correctness": correctness})
    rubric = Rubric(
        criteria=(Criterion(id="correctness", description="Correct."),)
    )

    with pytest.raises(TypeError, match="use ascore"):
        grader.score("42", rubric)


def test_ascore_supports_mixed_criterion_graders() -> None:
    async def run() -> None:
        async def correctness(answer: str, criterion: Criterion) -> Score:
            await asyncio.sleep(0)
            return Score(
                float(answer == "42"),
                metadata={"feedback": "correct"},
            )

        grader = CompositeGrader[str](
            {
                "correctness": correctness,
                "format": lambda answer, criterion: 0.5,
            }
        )
        score = await grader.ascore("42", answer_rubric())

        assert score.value == pytest.approx(2.125 / 2.25)
        assert score.component_values == {"correctness": 1.0, "format": 0.5}
        assert score.components["correctness"].metadata == {
            "feedback": "correct"
        }

    asyncio.run(run())


def test_composite_grader_requires_implementations_for_all_criteria() -> None:
    grader = CompositeGrader[str]({"correctness": lambda answer, criterion: 1.0})

    with pytest.raises(ValueError, match="format"):
        grader.score("42", answer_rubric())


def test_score_rejects_non_finite_values() -> None:
    with pytest.raises(ValueError, match="finite"):
        Score(float("nan"))


def test_score_is_recursive_and_serializes_through_info() -> None:
    score = Score(
        0.75,
        {"correctness": Score(1.0, feedback="Correct."), "style": 0.5},
        {"model": "judge"},
        feedback="Mostly correct.",
    )

    assert score.component_values == {"correctness": 1.0, "style": 0.5}
    assert Score.from_info(score.as_info()) == score
    assert Score.from_info({}, default=score) is score
    assert Score.from_info({}) is None


def test_additional_aggregation_strategies() -> None:
    rubric = Rubric(
        criteria=(
            Criterion(id="required", description="Required."),
            Criterion(id="bonus", description="Bonus.", kind="bonus"),
            Criterion(id="safe", description="Avoid unsafe content.", kind="penalty"),
        )
    )
    components = {
        "required": Score(1.0),
        "bonus": Score(0.5),
        "safe": Score(0.0),
    }

    assert all_pass(rubric, components) == 0.0
    assert asymmetric_mean(rubric, components) == pytest.approx(0.5)


def test_grader_protocol_is_generic_over_input_and_rubric() -> None:
    def correctness(answer: str, criterion: Criterion) -> float:
        return float(answer == "42" and criterion.id == "correctness")

    grader: Grader[str, Criterion] = correctness
    criterion = Criterion(id="correctness", description="Correct.")
    assert grader("42", criterion) == 1.0
