from __future__ import annotations

import asyncio
from typing import Any

import gymnasium as gym
import pytest

import rolloutlib
import rolloutlib.envs.language as legacy_language_wrappers
import rolloutlib.envs.wrappers as legacy_async_wrappers
from rolloutlib import spaces, wrappers
from rolloutlib.envs import AsyncEnv
from rolloutlib.types import Chat, ToolCall
from rolloutlib.wrappers import (
    AsyncActionWrapper,
    AsyncObservationWrapper,
    AsyncRewardWrapper,
)


def test_wrappers_namespace_is_canonical_and_legacy_exports_remain_compatible() -> None:
    assert rolloutlib.ChatObservationWrapper is wrappers.ChatObservationWrapper
    assert rolloutlib.GradingWrapper is wrappers.GradingWrapper
    assert legacy_language_wrappers.ChatObservationWrapper is (
        wrappers.ChatObservationWrapper
    )
    assert legacy_async_wrappers.AsyncWrapper is wrappers.AsyncWrapper
    assert rolloutlib.envs.ToolCallActionWrapper is wrappers.ToolCallActionWrapper
    assert rolloutlib.envs.AsyncGradingWrapper is wrappers.AsyncGradingWrapper


class ToolEnv(AsyncEnv[Chat, ToolCall]):
    action_space = spaces.tools.call(
        {
            "search": gym.spaces.Dict(
                {"query": spaces.text.text(min_length=1, max_length=20)}
            )
        }
    )
    observation_space = spaces.messages.chat(min_length=1, max_length=2)
    metadata = {"name": "tool-env"}

    def __init__(self) -> None:
        self.last_action: ToolCall | None = None
        self.closed = False

    async def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Chat, dict[str, Any]]:
        await super().reset(seed=seed, options=options)
        return [{"role": "user", "content": "search"}], {"reset": True}

    async def step(
        self, action: ToolCall
    ) -> tuple[Chat, float, bool, bool, dict[str, Any]]:
        assert action in self.action_space
        self.last_action = action
        return (
            [{"role": "tool", "content": "result"}],
            1.5,
            True,
            False,
            {},
        )

    async def close(self) -> None:
        self.closed = True


class ParseToolCall(AsyncActionWrapper[Chat, tuple[int, ...], ToolCall]):
    def __init__(self, env: ToolEnv) -> None:
        super().__init__(env)
        self.action_space = spaces.tokens.sequence(256)

    async def action(self, action: tuple[int, ...]) -> ToolCall:
        return {
            "name": "search",
            "arguments": {"query": "".join(map(chr, action))},
        }


class LastMessage(AsyncObservationWrapper[str, tuple[int, ...], Chat]):
    def __init__(self, env: ParseToolCall) -> None:
        super().__init__(env)
        self.observation_space = spaces.text.text(min_length=1, max_length=20)

    async def observation(self, observation: Chat) -> str:
        content = observation[-1]["content"]
        assert isinstance(content, str)
        return content


class DoubleReward(AsyncRewardWrapper[str, tuple[int, ...]]):
    async def reward(self, reward: float) -> float:
        return reward * 2


def test_async_wrappers_transform_values_and_preserve_space_direction() -> None:
    async def run() -> None:
        inner = ToolEnv()
        action_wrapper = ParseToolCall(inner)
        observation_wrapper = LastMessage(action_wrapper)
        env = DoubleReward(observation_wrapper)

        # The policy-facing space belongs to the outer action wrapper. The
        # semantic environment continues to advertise structured tool calls.
        assert action_wrapper.action_space is not inner.action_space
        assert (115, 101, 97, 114, 99, 104) in action_wrapper.action_space
        assert {
            "name": "search",
            "arguments": {"query": "search"},
        } in inner.action_space

        observation, info = await env.reset(seed=10)
        step_result = await env.step((115, 101, 97, 114, 99, 104))

        assert observation == "search"
        assert info == {"reset": True}
        assert step_result == ("result", 3.0, True, False, {})
        assert inner.last_action == {
            "name": "search",
            "arguments": {"query": "search"},
        }
        assert env.action_space is action_wrapper.action_space
        assert env.observation_space is observation_wrapper.observation_space
        assert env.metadata == {"name": "tool-env"}
        assert env.unwrapped is inner

        await env.close()
        assert inner.closed

    asyncio.run(run())


def test_async_action_wrapper_rejects_invalid_transformed_actions() -> None:
    async def run() -> None:
        wrapper = ParseToolCall(ToolEnv())
        oversized_query = tuple(ord("x") for _ in range(21))

        with pytest.raises(ValueError, match="transformed action.*outside"):
            await wrapper.step(oversized_query)

    asyncio.run(run())
