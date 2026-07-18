from __future__ import annotations

import asyncio

import gymnasium as gym
import pytest

from rolloutlib.graders import (
    AsyncCompositeGrader,
    AsyncGrader,
    AsyncRewardGrader,
    AsyncRubricGrader,
    CompositeGrader,
    Criterion,
    Grader,
    RewardGrader,
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


def test_rubric_grader_binds_a_rubric_and_validates_criterion_results() -> None:
    rubric = answer_rubric()
    grader = RubricGrader[str](
        rubric,
        lambda answer, rubric: {
            "correctness": Score(
                float(answer == "42"),
                feedback="The answer is correct.",
            ),
            "format": 0.5,
        },
        input_space=ANSWER_SPACE,
        metadata={"judge": "test"},
    )

    score = grader.grade("42")

    assert grader.rubric is rubric
    assert score.value == pytest.approx(2.125 / 2.25)
    assert score.component_values == {"correctness": 1.0, "format": 0.5}
    assert score.components["correctness"].feedback == "The answer is correct."
    assert score.metadata["judge"] == "test"
    assert score.metadata["rubric_id"] == "answer-quality"
    assert score.metadata["rubric_version"] == "1"


def test_rubric_grader_supports_custom_aggregation() -> None:
    rubric = answer_rubric()
    grader = RubricGrader[str](
        rubric,
        lambda answer, rubric: {
            "correctness": float(answer == "42"),
            "format": 0.5,
        },
        input_space=ANSWER_SPACE,
        aggregate=weighted_sum,
    )

    assert grader.grade("42").value == pytest.approx(2.125)


def test_rubric_grader_rejects_missing_and_unknown_criterion_results() -> None:
    rubric = answer_rubric()
    missing = RubricGrader(
        rubric,
        lambda answer, rubric: {"correctness": 1.0},
        input_space=ANSWER_SPACE,
    )
    unknown = RubricGrader(
        rubric,
        lambda answer, rubric: {
            "correctness": 1.0,
            "format": 1.0,
            "style": 1.0,
        },
        input_space=ANSWER_SPACE,
    )

    with pytest.raises(ValueError, match="omitted components: format"):
        missing.grade("42")
    with pytest.raises(ValueError, match="unknown components: style"):
        unknown.grade("42")


def test_rubric_grader_requires_a_mapping_of_criterion_results() -> None:
    grader = RubricGrader(
        answer_rubric(),
        lambda answer, rubric: 1.0,  # type: ignore[arg-type,return-value]
        input_space=ANSWER_SPACE,
    )

    with pytest.raises(TypeError, match="must return a mapping"):
        grader.grade("42")


def test_async_rubric_grader_accepts_a_sync_or_async_judge() -> None:
    async def run() -> None:
        rubric = answer_rubric()

        async def judge(answer: str, rubric: Rubric) -> dict[str, float]:
            await asyncio.sleep(0)
            return {"correctness": float(answer == "42"), "format": 1.0}

        async_grader = AsyncRubricGrader(
            rubric,
            judge,
            input_space=ANSWER_SPACE,
        )
        sync_grader = AsyncRubricGrader(
            rubric,
            lambda answer, rubric: {"correctness": 1.0, "format": 1.0},
            input_space=ANSWER_SPACE,
        )

        assert (await async_grader.grade("42")).value == 1.0
        assert (await sync_grader.grade("42")).value == 1.0

    asyncio.run(run())


def test_reward_grader_combines_named_reward_functions() -> None:
    grader = RewardGrader[str](
        {
            "correctness": lambda answer: Score(
                float(answer == "42"),
                feedback="Exact match.",
            ),
            "format": lambda answer: 0.5,
        },
        input_space=ANSWER_SPACE,
        weights={"correctness": 2.0, "format": 0.25},
        metadata={"suite": "answer"},
    )

    score = grader.grade("42")

    assert score.value == pytest.approx(2.125)
    assert score.component_values == {"correctness": 1.0, "format": 0.5}
    assert score.components["correctness"].feedback == "Exact match."
    assert score.metadata["suite"] == "answer"


def test_reward_grader_supports_custom_aggregation_and_validates_weights() -> None:
    gated = RewardGrader[str](
        {
            "quality": lambda answer: 0.75,
            "tests": lambda answer: 0.0,
        },
        input_space=ANSWER_SPACE,
        aggregate=lambda scores: (
            scores["quality"].value if scores["tests"].value == 1.0 else 0.0
        ),
    )

    assert gated.grade("42").value == 0.0

    with pytest.raises(ValueError, match="unknown components"):
        RewardGrader(
            {"correctness": lambda answer: 1.0},
            input_space=ANSWER_SPACE,
            weights={"other": 1.0},
        )


def test_async_reward_grader_runs_rewards_concurrently() -> None:
    async def run() -> None:
        release = asyncio.Event()
        both_started = asyncio.Event()
        started = 0

        async def reward(value: object) -> float:
            nonlocal started
            started += 1
            if started == 2:
                both_started.set()
            await release.wait()
            return 1.0

        grader = AsyncRewardGrader(
            {"first": reward, "second": reward},
            input_space=PydanticSpace(object),
        )
        evaluation = asyncio.create_task(grader.grade(object()))

        await asyncio.wait_for(both_started.wait(), timeout=1)
        assert not evaluation.done()
        release.set()
        score = await evaluation
        assert score.value == 2.0
        assert score.component_values == {"first": 1.0, "second": 1.0}

    asyncio.run(run())


def test_composite_grader_preserves_nested_scores() -> None:
    rubric_grader = RubricGrader[str](
        answer_rubric(),
        lambda answer, rubric: {
            "correctness": float(answer == "42"),
            "format": 0.5,
        },
        input_space=ANSWER_SPACE,
    )
    reward_grader = RewardGrader[str](
        {"tests": lambda answer: float(answer == "42")},
        input_space=ANSWER_SPACE,
    )
    grader = CompositeGrader[str](
        {
            "quality": rubric_grader,
            "verification": reward_grader,
        },
        input_space=ANSWER_SPACE,
        weights={"quality": 0.4, "verification": 0.6},
    )

    score = grader.grade("42")

    expected_quality = 2.125 / 2.25
    assert score.value == pytest.approx(0.4 * expected_quality + 0.6)
    assert score.component_values == {
        "quality": pytest.approx(expected_quality),
        "verification": 1.0,
    }
    assert score.components["quality"].component_values == {
        "correctness": 1.0,
        "format": 0.5,
    }
    assert score.components["verification"].component_values == {"tests": 1.0}


def test_async_composite_grader_accepts_sync_and_async_children() -> None:
    async def run() -> None:
        sync_grader = RewardGrader[str](
            {"exact": lambda answer: float(answer == "42")},
            input_space=ANSWER_SPACE,
        )

        async def slow(answer: str) -> float:
            await asyncio.sleep(0)
            return 0.5

        async_grader = AsyncRewardGrader[str](
            {"style": slow},
            input_space=ANSWER_SPACE,
        )
        grader = AsyncCompositeGrader[str](
            {"verification": sync_grader, "quality": async_grader},
            input_space=ANSWER_SPACE,
        )

        score = await grader.grade("42")

        assert score.value == 0.75
        assert score.component_values == {"verification": 1.0, "quality": 0.5}

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


def test_additional_rubric_aggregation_strategies() -> None:
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


def test_grader_contracts_are_nominal_and_enforce_input_spaces() -> None:
    called = False

    class ExactMatchGrader(Grader[str]):
        input_space = ANSWER_SPACE

        def _grade(self, input: str) -> Score:
            nonlocal called
            called = True
            return Score(float(input == "42"))

    class AsyncExactMatchGrader(AsyncGrader[str]):
        input_space = ANSWER_SPACE

        async def _grade(self, input: str) -> Score:
            return Score(float(input == "42"))

    grader: Grader[str] = ExactMatchGrader()
    async_grader: AsyncGrader[str] = AsyncExactMatchGrader()

    assert grader.grade("42") == Score(1.0)
    assert asyncio.run(async_grader.grade("42")) == Score(1.0)

    called = False
    with pytest.raises(ValueError, match="outside input_space"):
        grader.grade(42)  # type: ignore[arg-type]
    assert called is False


def test_composite_grader_rejects_incompatible_child_input_spaces() -> None:
    text_grader = RewardGrader(
        {"length": lambda value: float(bool(value))},
        input_space=gym.spaces.Text(max_length=20),
    )

    with pytest.raises(
        TypeError,
        match="composite input_space.*incompatible.*grader 'text' input_space",
    ):
        CompositeGrader(
            {"text": text_grader},  # type: ignore[dict-item]
            input_space=gym.spaces.Discrete(5),
        )
