from __future__ import annotations

import asyncio
from typing import Any

import gymnasium as gym
import pytest

from rolloutlib import spaces
from rolloutlib.envs import (
    Env,
    GradingWrapper,
    SingleTurnEnv,
    check_env,
)
from rolloutlib.graders import (
    RubricGrader,
    Criterion,
    Rubric,
    Score,
)


class EchoEnv(gym.Env[str, int]):
    action_space = gym.spaces.Discrete(10)
    observation_space = gym.spaces.Text(min_length=0, max_length=20)

    def __init__(self) -> None:
        super().__init__()
        self.closed = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        super().reset(seed=seed)
        return "ready", {"options": options}

    def step(self, action: int) -> tuple[str, float, bool, bool, dict[str, Any]]:
        return str(action), float(action), True, False, {"action": action}

    def close(self) -> None:
        self.closed += 1


class CountingEnv(Env[int, int]):
    action_space = gym.spaces.Discrete(10)
    observation_space = gym.spaces.Discrete(100)

    def __init__(self) -> None:
        self.value = 0
        self.closed = False

    async def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        super().reset(seed=seed, options=options)
        self.value = int((options or {}).get("value", 0))
        return self.value, {}

    async def step(self, action: int) -> tuple[int, float, bool, bool, dict[str, Any]]:
        self.value += action
        return self.value, float(action), self.value >= 5, False, {}

    async def close(self) -> None:
        self.closed = True


def test_existing_sync_gymnasium_environment_remains_usable() -> None:
    env = EchoEnv()

    observation, info = env.reset(seed=11, options={"source": "test"})
    step_result = env.step(3)

    assert isinstance(env, gym.Env)
    assert observation in env.observation_space
    assert info == {"options": {"source": "test"}}
    assert step_result == ("3", 3.0, True, False, {"action": 3})


def test_async_env_uses_gymnasium_five_tuple_and_seed_semantics() -> None:
    async def run() -> None:
        env = CountingEnv()
        observation, info = await env.reset(seed=42, options={"value": 2})
        step_result = await env.step(3)

        assert observation == 2
        assert info == {}
        assert step_result == (5, 3.0, True, False, {})
        assert len(step_result) == 5
        assert env.np_random_seed == 42

    asyncio.run(run())


def test_async_environment_conformance_check() -> None:
    async def run() -> None:
        await check_env(CountingEnv(), action=0)

    asyncio.run(run())



class OneTurn(SingleTurnEnv[str, int]):
    action_space = gym.spaces.Discrete(10)
    observation_space = spaces.TextSpace(max_length=20)

    def initial_observation(
        self, *, options: dict[str, Any] | None = None
    ) -> tuple[str, dict[str, Any]]:
        return "question", {}

    def evaluate(self, action: int) -> tuple[float, dict[str, Any]]:
        return float(action == 7), {"answer": action}

    def terminal_observation(self, action: int) -> str:
        return f"answer:{action}"


class AwaitableOneTurn(SingleTurnEnv[str, int]):
    action_space = gym.spaces.Discrete(10)
    observation_space = spaces.TextSpace(max_length=20)

    async def initial_observation(
        self, *, options: dict[str, Any] | None = None
    ) -> tuple[str, dict[str, Any]]:
        return "question", {}

    async def evaluate(self, action: int) -> tuple[float, dict[str, Any]]:
        await asyncio.sleep(0)
        return float(action == 7), {"answer": action}

    async def terminal_observation(self, action: int) -> str:
        return f"answer:{action}"


def test_sync_single_turn_environment_owns_one_step_lifecycle() -> None:
    env = OneTurn()

    assert env.reset() == ("question", {})
    assert env.step(7) == ("answer:7", 1.0, True, False, {"answer": 7})
    with pytest.raises(RuntimeError, match=r"reset\(\) must be called"):
        env.step(7)


def test_async_single_turn_environment_matches_sync_semantics() -> None:
    async def run() -> None:
        env = AwaitableOneTurn()

        assert await env.reset(seed=123) == ("question", {})
        assert await env.step(7) == (
            "answer:7",
            1.0,
            True,
            False,
            {"answer": 7},
        )
        with pytest.raises(RuntimeError, match=r"reset\(\) must be called"):
            await env.step(7)

    asyncio.run(run())


def test_single_turn_environment_preserves_a_structured_score() -> None:
    class ScoredOneTurn(SingleTurnEnv[str, int]):
        action_space = gym.spaces.Discrete(10)
        observation_space = spaces.TextSpace(max_length=20)

        def initial_observation(
            self, *, options: dict[str, Any] | None = None
        ) -> tuple[str, dict[str, Any]]:
            return "question", {}

        def evaluate(self, action: int) -> Score:
            return Score(
                float(action == 7),
                {"correct": float(action == 7)},
                feedback="Checked the answer.",
            )

        def terminal_observation(self, action: int) -> str:
            return f"answer:{action}"

    env = ScoredOneTurn()
    env.reset()
    _, reward, terminated, _, info = env.step(7)

    assert reward == 1.0
    assert terminated is True
    assert Score.from_info(info) == Score(
        1.0,
        {"correct": 1.0},
        feedback="Checked the answer.",
    )


def test_sync_grading_wrapper_grades_inside_step() -> None:
    rubric = Rubric(
        id="echo",
        criteria=(Criterion(id="correct", description="The action is three."),),
    )
    env = GradingWrapper(
        EchoEnv(),
        grader=RubricGrader(
            rubric,
            lambda action, rubric: {
                "correct": Score(float(action == 3)),
            },
            input_space=gym.spaces.Discrete(10),
        ),
    )

    env.reset()
    _, reward, terminated, _, info = env.step(3)
    score = Score.from_info(info)

    assert reward == 1.0
    assert terminated is True
    assert score is not None
    assert score.component_values == {"correct": 1.0}
    assert score.metadata["rubric_id"] == "echo"


def test_async_grading_wrapper_awaits_grading_inside_step() -> None:
    async def run() -> None:
        async def judge(action: int, rubric: Rubric) -> dict[str, Score]:
            await asyncio.sleep(0)
            return {"correct": Score(float(action == 3))}

        rubric = Rubric(
            criteria=(Criterion(id="correct", description="The value reaches five."),)
        )
        env = GradingWrapper(
            CountingEnv(),
            grader=RubricGrader(
                rubric,
                judge,
                input_space=gym.spaces.Discrete(10),
            ),
        )
        await env.reset(options={"value": 2})
        _, reward, terminated, _, info = await env.step(3)

        assert reward == 1.0
        assert terminated is True
        assert Score.from_info(info) is not None

    asyncio.run(run())


def test_grading_wrapper_rejects_incompatible_action_and_grader_spaces() -> None:
    grader = RubricGrader(
        Rubric(criteria=(Criterion(id="correct", description="Correct."),)),
        lambda value, rubric: {"correct": Score(1.0)},
        input_space=gym.spaces.Text(max_length=20),
    )

    with pytest.raises(
        TypeError,
        match="environment action_space.*incompatible.*grader input_space",
    ):
        GradingWrapper(EchoEnv(), grader=grader)


def test_grading_wrapper_requires_and_checks_custom_input_space() -> None:
    grader = RubricGrader(
        Rubric(criteria=(Criterion(id="correct", description="Correct."),)),
        lambda value, rubric: {"correct": Score(float(value == "3"))},
        input_space=gym.spaces.Text(max_length=20),
    )

    with pytest.raises(TypeError, match="input_space is required"):
        GradingWrapper(
            EchoEnv(),
            grader=grader,
            make_input=lambda environment, action: str(action),
        )

    env = GradingWrapper(
        EchoEnv(),
        grader=grader,
        make_input=lambda environment, action: str(action),
        input_space=gym.spaces.Text(max_length=10),
    )
    env.reset()
    _, reward, _, _, _ = env.step(3)

    assert reward == 1.0


def test_grading_wrapper_checks_custom_input_values_at_runtime() -> None:
    grader = RubricGrader(
        Rubric(criteria=(Criterion(id="correct", description="Correct."),)),
        lambda value, rubric: {"correct": Score(1.0)},
        input_space=gym.spaces.Text(max_length=20),
    )
    env = GradingWrapper(
        EchoEnv(),
        grader=grader,
        make_input=lambda environment, action: action,  # type: ignore[arg-type]
        input_space=gym.spaces.Text(max_length=10),
    )
    env.reset()

    with pytest.raises(ValueError, match="make_input result.*outside"):
        env.step(3)
