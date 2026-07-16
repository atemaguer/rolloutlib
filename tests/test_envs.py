from __future__ import annotations

import asyncio
import threading
import time
from typing import Any

import gymnasium as gym
import pytest
from gymnasium.utils.env_checker import check_env

from rolloutlib.envs import (
    AsyncEnv,
    AsyncFromSync,
    AsyncGradingWrapper,
    AsyncSingleTurnEnv,
    GradingWrapper,
    SingleTurnEnv,
    SyncFromAsync,
    as_async,
    as_sync,
    check_async_env,
)
from rolloutlib.graders import Criterion, Rubric, Score


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


class CountingAsyncEnv(AsyncEnv[int, int]):
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
        await super().reset(seed=seed, options=options)
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
        env = CountingAsyncEnv()
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
        await check_async_env(CountingAsyncEnv(), action=0)

    asyncio.run(run())


def test_as_async_is_identity_for_async_env_and_lifts_gym_env() -> None:
    native = CountingAsyncEnv()
    sync = EchoEnv()

    assert as_async(native) is native
    assert isinstance(as_async(sync), AsyncFromSync)

    with pytest.raises(TypeError, match="expected rolloutlib.envs.AsyncEnv"):
        as_async(object())  # type: ignore[arg-type]


def test_lifted_sync_calls_do_not_block_the_event_loop() -> None:
    started = threading.Event()
    release = threading.Event()

    class BlockingEnv(EchoEnv):
        def step(self, action: int) -> tuple[str, float, bool, bool, dict[str, Any]]:
            started.set()
            assert release.wait(timeout=2)
            return super().step(action)

    async def run() -> None:
        env = as_async(BlockingEnv())
        task = asyncio.create_task(env.step(4))
        assert await asyncio.to_thread(started.wait, 1)

        # Reaching this assertion while the synchronous call is still waiting
        # proves that the event-loop thread was not occupied by env.step().
        await asyncio.sleep(0)
        assert not task.done()

        release.set()
        assert await task == ("4", 4.0, True, False, {"action": 4})

    asyncio.run(run())


def test_lifted_sync_calls_are_serialized_per_environment() -> None:
    class SerializedEnv(EchoEnv):
        def __init__(self) -> None:
            super().__init__()
            self.active = 0
            self.max_active = 0
            self.state_lock = threading.Lock()

        def step(self, action: int) -> tuple[str, float, bool, bool, dict[str, Any]]:
            with self.state_lock:
                self.active += 1
                self.max_active = max(self.max_active, self.active)
            time.sleep(0.03)
            with self.state_lock:
                self.active -= 1
            return super().step(action)

    async def run() -> None:
        sync = SerializedEnv()
        env = as_async(sync)

        first, second = await asyncio.gather(env.step(1), env.step(2))

        assert first[0] == "1"
        assert second[0] == "2"
        assert sync.max_active == 1

    asyncio.run(run())


def test_lifted_close_is_idempotent_and_prevents_future_calls() -> None:
    async def run() -> None:
        sync = EchoEnv()
        env = as_async(sync)

        await env.close()
        await env.close()

        assert sync.closed == 1
        with pytest.raises(RuntimeError, match="environment is closed"):
            await env.reset()

    asyncio.run(run())


def test_as_sync_is_identity_for_gym_env_and_lifts_async_env() -> None:
    sync = EchoEnv()
    native = CountingAsyncEnv()

    assert as_sync(sync) is sync
    adapted = as_sync(native)
    try:
        assert isinstance(adapted, SyncFromAsync)
        assert isinstance(adapted, gym.Env)
    finally:
        adapted.close()

    with pytest.raises(TypeError, match="expected gymnasium.Env"):
        as_sync(object())  # type: ignore[arg-type]


def test_sync_adapter_uses_one_persistent_loop_across_lifecycle() -> None:
    class LoopBoundEnv(AsyncEnv[int, int]):
        action_space = gym.spaces.Discrete(2)
        observation_space = gym.spaces.Discrete(2)

        def __init__(self) -> None:
            self.loop_ids: list[int] = []
            self.thread_ids: list[int] = []
            self.close_count = 0
            self.loop_bound_lock: asyncio.Lock | None = None

        def record_context(self) -> None:
            self.loop_ids.append(id(asyncio.get_running_loop()))
            self.thread_ids.append(threading.get_ident())

        async def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
        ) -> tuple[int, dict[str, Any]]:
            await super().reset(seed=seed, options=options)
            self.record_context()
            self.loop_bound_lock = asyncio.Lock()
            return 0, {}

        async def step(
            self, action: int
        ) -> tuple[int, float, bool, bool, dict[str, Any]]:
            self.record_context()
            assert self.loop_bound_lock is not None
            async with self.loop_bound_lock:
                await asyncio.sleep(0)
            return action, 1.0, True, False, {}

        async def close(self) -> None:
            self.record_context()
            self.close_count += 1

    native = LoopBoundEnv()
    env = as_sync(native)

    assert env.reset(seed=5) == (0, {})
    assert env.step(1) == (1, 1.0, True, False, {})
    env.close()
    env.close()

    assert len(set(native.loop_ids)) == 1
    assert len(set(native.thread_ids)) == 1
    assert native.close_count == 1

    with pytest.raises(RuntimeError, match="environment is closed"):
        env.reset()
    with pytest.raises(RuntimeError, match="environment is closed"):
        env.step(1)


def test_sync_adapter_passes_gymnasium_env_checker() -> None:
    env = as_sync(CountingAsyncEnv())
    try:
        check_env(env, skip_render_check=True)
    finally:
        env.close()


class OneTurn(SingleTurnEnv[str, int]):
    action_space = gym.spaces.Discrete(10)
    observation_space = gym.spaces.Text(min_length=0, max_length=20)

    def initial_observation(
        self, *, options: dict[str, Any] | None = None
    ) -> tuple[str, dict[str, Any]]:
        return "question", {}

    def evaluate(self, action: int) -> tuple[float, dict[str, Any]]:
        return float(action == 7), {"answer": action}

    def terminal_observation(self, action: int) -> str:
        return f"answer:{action}"


class AsyncOneTurn(AsyncSingleTurnEnv[str, int]):
    action_space = gym.spaces.Discrete(10)
    observation_space = gym.spaces.Text(min_length=0, max_length=20)

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
        env = AsyncOneTurn()

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
        observation_space = gym.spaces.Text(min_length=0, max_length=20)

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
        rubric=rubric,
        grader=lambda action, _: Score(
            float(action == 3),
            {"correct": float(action == 3)},
        ),
        make_input=lambda environment, action: action,
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
        async def grade(action: int, rubric: Rubric) -> Score:
            await asyncio.sleep(0)
            return Score(float(action == 3), {"correct": float(action == 3)})

        rubric = Rubric(
            criteria=(Criterion(id="correct", description="The value reaches five."),)
        )
        env = AsyncGradingWrapper(
            CountingAsyncEnv(),
            rubric=rubric,
            grader=grade,
            make_input=lambda environment, action: action,
        )
        await env.reset(options={"value": 2})
        _, reward, terminated, _, info = await env.step(3)

        assert reward == 1.0
        assert terminated is True
        assert Score.from_info(info) is not None

    asyncio.run(run())
