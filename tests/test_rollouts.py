from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

import gymnasium as gym
import pytest

from rolloutlib import spaces
from rolloutlib.envs import Env
from rolloutlib.graders import Score
from rolloutlib.rollouts import (
    PolicyOutput,
    arollout,
    arollout_group,
    rollout,
    rollout_group,
)


class CountingEnv(gym.Env[int, int]):
    action_space = gym.spaces.Discrete(2)
    observation_space = gym.spaces.Discrete(4)

    def __init__(self, target: int = 3) -> None:
        super().__init__()
        self.target = target
        self.value = 0
        self.closed = False

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        super().reset(seed=seed)
        del options
        self.value = 0
        return self.value, {"target": self.target}

    def step(self, action: int) -> tuple[int, float, bool, bool, dict[str, Any]]:
        self.value += action + 1
        return (
            self.value,
            float(action),
            self.value >= self.target,
            False,
            {"value": self.value},
        )

    def close(self) -> None:
        self.closed = True


class AwaitableCountingEnv(Env[int, int]):
    action_space = gym.spaces.Discrete(2)
    observation_space = gym.spaces.Discrete(4)

    def __init__(self, target: int = 3) -> None:
        self.target = target
        self.value = 0
        self.closed = False

    async def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        super().reset(seed=seed, options=options)
        self.value = 0
        return self.value, {"target": self.target}

    async def step(self, action: int) -> tuple[int, float, bool, bool, dict[str, Any]]:
        self.value += action + 1
        return (
            self.value,
            float(action),
            self.value >= self.target,
            False,
            {"value": self.value},
        )

    async def close(self) -> None:
        self.closed = True


class ScoredCountingEnv(CountingEnv):
    def step(self, action: int) -> tuple[int, float, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = super().step(action)
        if terminated:
            score = Score(
                1.0,
                {"correct": 1.0},
                feedback="Environment score.",
            )
            info.update(score.as_info())
            reward = score.value
        return observation, reward, terminated, truncated, info


def test_rollout_records_policy_and_gymnasium_steps() -> None:
    environment = CountingEnv()

    trajectory = rollout(
        environment,
        lambda _: PolicyOutput(0, {"tokens": [0]}),
        metadata={"source": "test"},
    )

    assert trajectory.initial_observation == 0
    assert trajectory.initial_info == {"target": 3}
    assert trajectory.observations == (0, 1, 2, 3)
    assert trajectory.actions == (0, 0, 0)
    assert trajectory.rewards == (0.0, 0.0, 0.0)
    assert trajectory.total_reward == 0.0
    assert trajectory.terminated is True
    assert trajectory.truncated is False
    assert trajectory.complete is True
    assert trajectory.steps[0].policy_info == {"tokens": [0]}
    assert trajectory.steps[0].policy_tokens == (0,)
    assert trajectory.metadata == {"source": "test"}
    assert environment.closed is False


def test_rollout_preserves_structured_policy_sampling_fields() -> None:
    output = PolicyOutput(
        action=0,
        tokens=[3, 4],
        logprobs=[-0.3, -0.4],
        stop_reason="length",
    )

    trajectory = rollout(CountingEnv(), lambda _: output)

    step = trajectory.steps[0]
    assert step.policy_tokens == (3, 4)
    assert step.policy_logprobs == (-0.3, -0.4)
    assert step.policy_stop_reason == "length"
    assert step.policy_info == {
        "tokens": (3, 4),
        "logprobs": (-0.3, -0.4),
        "stop_reason": "length",
    }


def test_rollout_max_steps_is_a_collection_truncation() -> None:
    trajectory = rollout(CountingEnv(target=10), lambda _: 0, max_steps=2)

    assert len(trajectory) == 2
    assert trajectory.complete is True
    assert trajectory.terminated is False
    assert trajectory.truncated is True


def test_rollout_rejects_policy_actions_outside_environment_space() -> None:
    environment = CountingEnv()

    with pytest.raises(ValueError, match="policy action.*outside"):
        rollout(environment, lambda observation: 3)

    assert environment.value == 0


def test_rollout_rejects_incompatible_declared_policy_spaces_before_reset() -> None:
    class DeclaredPolicy:
        observation_space = gym.spaces.Discrete(4)
        action_space = gym.spaces.Text(max_length=10)

        def __call__(self, observation: int) -> str:
            return "invalid"

    environment = CountingEnv()

    with pytest.raises(
        TypeError,
        match="policy action_space.*incompatible.*environment action_space",
    ):
        rollout(environment, DeclaredPolicy())  # type: ignore[arg-type]

    assert environment.value == 0


def test_rollout_group_closes_envs_and_preserves_environment_scores() -> None:
    environments: list[ScoredCountingEnv] = []

    def make_env(item: int) -> ScoredCountingEnv:
        environment = ScoredCountingEnv(target=item)
        environments.append(environment)
        return environment

    group = rollout_group(
        3,
        make_env,
        lambda _: 0,
        num_rollouts=2,
        item_id="example-3",
    )

    assert group.item == 3
    assert group.item_id == "example-3"
    assert len(group.trajectories) == 2
    assert group.rewards == (1.0, 1.0)
    assert group.scores[0].component_values == {"correct": 1.0}
    assert group.scores[0].feedback == "Environment score."
    assert group.trajectories[0].steps[-1].score == group.scores[0]
    assert all(trajectory.complete for trajectory in group.trajectories)
    assert all(environment.closed for environment in environments)


def test_async_rollout_group_supports_async_policies_and_bounded_concurrency() -> None:
    async def run() -> None:
        environment = AwaitableCountingEnv()
        trajectory = await arollout(environment, lambda _: 0)
        assert trajectory.complete is True
        assert trajectory.terminated is True
        await environment.close()

        environments: list[AwaitableCountingEnv] = []

        def make_env(item: int) -> AwaitableCountingEnv:
            environment = AwaitableCountingEnv(target=item)
            environments.append(environment)
            return environment

        async def policy(_: int) -> int:
            return 0

        group = await arollout_group(
            3,
            make_env,
            policy,
            num_rollouts=3,
            concurrency=2,
        )
        assert len(group.trajectories) == 3
        assert all(trajectory.complete for trajectory in group.trajectories)
        assert all(environment.closed for environment in environments)

    asyncio.run(run())


def test_group_requires_positive_rollout_count() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        rollout_group(1, CountingEnv, lambda _: 0, num_rollouts=0)


def test_group_requires_exactly_one_scalar_or_batch_policy() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        rollout_group(1, CountingEnv)

    with pytest.raises(ValueError, match="exactly one"):
        rollout_group(
            1,
            CountingEnv,
            lambda _: 0,
            batch_policy=lambda observations: [0] * len(observations),
        )

    async def run() -> None:
        with pytest.raises(ValueError, match="exactly one"):
            await arollout_group(1, AwaitableCountingEnv)

        with pytest.raises(ValueError, match="exactly one"):
            await arollout_group(
                1,
                AwaitableCountingEnv,
                lambda _: 0,
                batch_policy=lambda observations: [0] * len(observations),
            )

    asyncio.run(run())


def test_rollout_group_batches_policy_calls_and_preserves_slot_data() -> None:
    environments: list[CountingEnv] = []
    batch_sizes: list[int] = []

    def make_env(item: int) -> CountingEnv:
        environment = CountingEnv(target=item)
        environments.append(environment)
        return environment

    def policy(observations: Sequence[int]) -> list[PolicyOutput[int]]:
        batch_sizes.append(len(observations))
        return [
            PolicyOutput(
                action=0,
                info={"slot": index},
                tokens=[observation],
                logprobs=[-0.1],
            )
            for index, observation in enumerate(observations)
        ]

    group = rollout_group(
        2,
        make_env,
        batch_policy=policy,
        num_rollouts=3,
        item_id="vector-2",
    )

    assert group.item_id == "vector-2"
    assert batch_sizes == [3, 3]
    assert [trajectory.observations for trajectory in group.trajectories] == [
        (0, 1, 2),
        (0, 1, 2),
        (0, 1, 2),
    ]
    assert [trajectory.initial_info for trajectory in group.trajectories] == [
        {"target": 2},
        {"target": 2},
        {"target": 2},
    ]
    assert group.trajectories[1].steps[0].policy_info == {
        "slot": 1,
        "tokens": (0,),
        "logprobs": (-0.1,),
    }
    assert all(environment.closed for environment in environments)


def test_rollout_group_handles_uneven_batch_episode_lengths() -> None:
    environments: list[CountingEnv] = []
    batch_sizes: list[int] = []

    def make_env(_: None) -> CountingEnv:
        environment = CountingEnv(target=len(environments) + 1)
        environments.append(environment)
        return environment

    def policy(observations: Sequence[int]) -> list[int]:
        batch_sizes.append(len(observations))
        return [0] * len(observations)

    group = rollout_group(
        None,
        make_env,
        batch_policy=policy,
        num_rollouts=2,
    )

    assert batch_sizes == [2, 1]
    assert [len(trajectory) for trajectory in group.trajectories] == [1, 2]
    assert all(environment.closed for environment in environments)


def test_rollout_group_batches_language_spaces() -> None:
    class LanguageEnv(gym.Env[Any, str]):
        observation_space = spaces.messages.chat(min_length=1)
        action_space = spaces.Text(min_length=1)

        def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
        ) -> tuple[Any, dict[str, Any]]:
            super().reset(seed=seed)
            del options
            return [{"role": "user", "content": "Choose an action."}], {}

        def step(
            self,
            action: str,
        ) -> tuple[Any, float, bool, bool, dict[str, Any]]:
            observation = [{"role": "assistant", "content": action}]
            return observation, 1.0, True, False, {"answer": action}

    group = rollout_group(
        None,
        lambda _: LanguageEnv(),
        batch_policy=lambda observations: ["move"] * len(observations),
        num_rollouts=3,
    )

    assert [trajectory.actions for trajectory in group.trajectories] == [
        ("move",),
        ("move",),
        ("move",),
    ]
    assert [trajectory.steps[0].info for trajectory in group.trajectories] == [
        {"answer": "move"},
        {"answer": "move"},
        {"answer": "move"},
    ]


def test_rollout_group_validates_batch_size_before_step() -> None:
    environments: list[CountingEnv] = []

    def make_env(item: int) -> CountingEnv:
        environment = CountingEnv(target=item)
        environments.append(environment)
        return environment

    with pytest.raises(ValueError, match="1 outputs for 2 observations"):
        rollout_group(
            2,
            make_env,
            batch_policy=lambda _: [0],
            num_rollouts=2,
        )

    assert all(environment.value == 0 for environment in environments)
    assert all(environment.closed for environment in environments)


def test_rollout_group_closes_partial_batch_when_factory_fails() -> None:
    environments: list[CountingEnv] = []

    def make_env(_: None) -> CountingEnv:
        if environments:
            raise RuntimeError("factory failed")
        environment = CountingEnv()
        environments.append(environment)
        return environment

    with pytest.raises(RuntimeError, match="factory failed"):
        rollout_group(
            None,
            make_env,
            batch_policy=lambda observations: [0] * len(observations),
            num_rollouts=2,
        )

    assert environments[0].closed


def test_arollout_group_shrinks_active_batches() -> None:
    async def run() -> None:
        environments: list[AwaitableCountingEnv] = []
        batch_sizes: list[int] = []

        def make_env(_: None) -> AwaitableCountingEnv:
            environment = AwaitableCountingEnv(target=len(environments) + 1)
            environments.append(environment)
            return environment

        async def policy(observations: Sequence[int]) -> list[PolicyOutput[int]]:
            batch_sizes.append(len(observations))
            return [
                PolicyOutput(
                    action=0,
                    info={"active_index": index},
                    tokens=[observation],
                    stop_reason="stop",
                )
                for index, observation in enumerate(observations)
            ]

        group = await arollout_group(
            None,
            make_env,
            batch_policy=policy,
            num_rollouts=3,
        )

        assert batch_sizes == [3, 2, 1]
        assert [len(trajectory) for trajectory in group.trajectories] == [1, 2, 3]
        assert [trajectory.terminated for trajectory in group.trajectories] == [
            True,
            True,
            True,
        ]
        assert group.trajectories[2].steps[1].policy_tokens == (1,)
        assert group.trajectories[2].steps[1].policy_stop_reason == "stop"
        assert all(environment.closed for environment in environments)

    asyncio.run(run())


def test_arollout_group_validates_batch_actions_before_any_active_step() -> None:
    async def run() -> None:
        environments: list[AwaitableCountingEnv] = []

        def make_env(_: None) -> AwaitableCountingEnv:
            environment = AwaitableCountingEnv(target=1)
            environments.append(environment)
            return environment

        with pytest.raises(ValueError, match="policy action.*outside"):
            await arollout_group(
                None,
                make_env,
                batch_policy=lambda observations: [0, 3][: len(observations)],
                num_rollouts=2,
            )

        assert all(environment.value == 0 for environment in environments)
        assert all(environment.closed for environment in environments)

    asyncio.run(run())


def test_arollout_group_bounds_batched_async_environment_creation() -> None:
    async def run() -> None:
        active_creations = 0
        maximum_creations = 0

        async def make_env(_: None) -> AwaitableCountingEnv:
            nonlocal active_creations, maximum_creations
            active_creations += 1
            maximum_creations = max(maximum_creations, active_creations)
            await asyncio.sleep(0)
            active_creations -= 1
            return AwaitableCountingEnv(target=1)

        await arollout_group(
            None,
            make_env,
            batch_policy=lambda observations: [0] * len(observations),
            num_rollouts=4,
            concurrency=2,
        )

        assert maximum_creations == 2

    asyncio.run(run())
