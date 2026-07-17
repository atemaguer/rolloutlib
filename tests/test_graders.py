from __future__ import annotations

import asyncio

import gymnasium as gym
import pytest

from rolloutlib.graders import (
    AsyncCallableGrader,
    AsyncGrader,
    AsyncRubricGrader,
    Criterion,
    CallableGrader,
    Grader,
    Rubric,
    RubricGrader,
    Score,
    all_pass,
    asymmetric_mean,
    weighted_sum,
)
from rolloutlib.spaces import PydanticSpace


ANSWER_SPACE = gym.spaces.Text(max_length=100)


def answer_rubric() -> Rubric:
    return Rubric(
        id="answer-quality",
        version="1",
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
    )


def test_callable_grader_normalizes_results_and_supports_binding() -> None:
    grader = CallableGrader[str](
        lambda answer, rubric: float(answer == "42"),
        input_space=ANSWER_SPACE,
        metadata={"implementation": "exact-match"},
    )
    rubric = answer_rubric()

    direct = grader.grade("42", rubric=rubric)
    bound = grader.bind(rubric).grade("42")

    assert direct == bound
    assert direct.value == 1.0
    assert direct.metadata["implementation"] == "exact-match"
    assert direct.metadata["rubric_id"] == "answer-quality"
    assert direct.metadata["rubric_version"] == "1"

    with pytest.raises(ValueError, match="another rubric"):
        grader.bind(rubric).grade(
            "42",
            rubric=Rubric(criteria=(Criterion(id="other", description="Other."),)),
        )


def test_rubric_grader_scores_each_criterion() -> None:
    grader = RubricGrader[str](
        lambda answer, criterion: 0.5 if answer.isdigit() else 0.0,
        input_space=ANSWER_SPACE,
        overrides={
            "correctness": lambda answer, criterion: Score(
                float(answer == "42"),
                feedback="The answer is correct.",
            ),
        },
        metadata={"grader": "local"},
    )

    score = grader.grade("42", rubric=answer_rubric())

    assert score.value == pytest.approx(2.125 / 2.25)
    assert score.component_values == {"correctness": 1.0, "format": 0.5}
    assert score.components["correctness"].feedback == "The answer is correct."
    assert score.metadata["grader"] == "local"


def test_rubric_grader_supports_custom_aggregation() -> None:
    grader = RubricGrader[str](
        {
            "correctness": lambda answer, criterion: float(answer == "42"),
            "format": lambda answer, criterion: 0.5,
        },
        input_space=ANSWER_SPACE,
        aggregate=weighted_sum,
    )

    assert grader.grade("42", rubric=answer_rubric()).value == pytest.approx(2.125)


def test_rubric_grader_requires_a_rubric_and_all_scorers() -> None:
    grader = RubricGrader[str](
        {"correctness": lambda answer, criterion: float(answer == "42")},
        input_space=ANSWER_SPACE,
    )

    with pytest.raises(ValueError, match="requires a rubric"):
        grader.grade("42")
    with pytest.raises(ValueError, match="format"):
        grader.grade("42", rubric=answer_rubric())


def test_async_callable_grader_accepts_sync_and_async_callables() -> None:
    async def run() -> None:
        async def exact(answer: str, rubric: Rubric | None) -> float:
            await asyncio.sleep(0)
            return float(answer == "42")

        async_grader = AsyncCallableGrader(exact, input_space=ANSWER_SPACE)
        sync_grader = AsyncCallableGrader(
            lambda answer, rubric: float(answer == "42"),
            input_space=ANSWER_SPACE,
        )

        assert (await async_grader.grade("42")).value == 1.0
        assert (await sync_grader.grade("42")).value == 1.0

    asyncio.run(run())


def test_async_rubric_grader_runs_criteria_concurrently() -> None:
    async def run() -> None:
        release = asyncio.Event()
        both_started = asyncio.Event()
        started = 0

        async def scorer(value: object, criterion: Criterion) -> float:
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
        grader = AsyncRubricGrader(
            {"first": scorer, "second": scorer},
            input_space=PydanticSpace(object),
        )
        evaluation = asyncio.create_task(grader.grade(object(), rubric=rubric))

        await asyncio.wait_for(both_started.wait(), timeout=1)
        assert not evaluation.done()
        release.set()
        score = await evaluation
        assert score.value == 1.0
        assert score.component_values == {"first": 1.0, "second": 1.0}

    asyncio.run(run())


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

    with pytest.raises(ValueError, match="finite"):
        Score(float("nan"))


def test_additional_aggregation_strategies() -> None:
    rubric = Rubric(
        criteria=(
            Criterion(id="required", description="Required."),
            Criterion(id="bonus", description="Bonus.", category="bonus"),
            Criterion(
                id="safe",
                description="Avoid unsafe content.",
                category="penalty",
            ),
        )
    )
    components = {
        "required": Score(1.0),
        "bonus": Score(0.5),
        "safe": Score(0.0),
    }

    assert all_pass(rubric, components) == 0.0
    assert asymmetric_mean(rubric, components) == pytest.approx(0.5)


def test_grader_contracts_are_nominal_and_return_scores() -> None:
    grader: Grader[str] = CallableGrader(
        lambda answer, rubric: float(answer == "42"),
        input_space=ANSWER_SPACE,
    )
    async_grader: AsyncGrader[str] = AsyncCallableGrader(
        lambda answer, rubric: float(answer == "42"),
        input_space=ANSWER_SPACE,
    )

    assert grader.grade("42") == Score(1.0)
    assert asyncio.run(async_grader.grade("42")) == Score(1.0)


def test_graders_enforce_their_input_space_before_evaluation() -> None:
    called = False

    def grade(answer: str, rubric: Rubric | None) -> float:
        nonlocal called
        called = True
        return 1.0

    grader = CallableGrader(grade, input_space=ANSWER_SPACE)

    with pytest.raises(ValueError, match="outside input_space"):
        grader.grade(42)  # type: ignore[arg-type]

    assert called is False
    assert grader.bind(answer_rubric()).input_space is ANSWER_SPACE


def test_custom_graders_can_declare_input_space_on_the_class() -> None:
    class ExactMatchGrader(Grader[str]):
        input_space = ANSWER_SPACE

        def _grade(
            self,
            input: str,
            *,
            rubric: Rubric | None = None,
        ) -> Score:
            return Score(float(input == "42"))

    grader = ExactMatchGrader()

    assert grader.grade("42") == Score(1.0)
    with pytest.raises(ValueError, match="outside input_space"):
        grader.grade("")  # The declared space requires at least one character.
