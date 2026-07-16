"""Composable wrappers for synchronous and asynchronous environments."""

from __future__ import annotations

import inspect
from abc import abstractmethod
from collections.abc import Callable
from typing import Any, Generic, TypeVar, cast

import gymnasium as gym
from gymnasium.spaces import Space

from ..graders import Grader, Rubric, Score
from .core import AsyncEnv


ObsT = TypeVar("ObsT")
ActT = TypeVar("ActT")
WrapperObsT = TypeVar("WrapperObsT")
WrapperActT = TypeVar("WrapperActT")
InputT = TypeVar("InputT")
RubricT = TypeVar("RubricT")


def _terminal(terminated: bool, truncated: bool) -> bool:
    return terminated or truncated


def _replace_reward(reward: float, score: Score) -> float:
    del reward
    return score.value


def _with_rubric_metadata(score: Score, rubric: object) -> Score:
    if not isinstance(rubric, Rubric):
        return score
    metadata = dict(score.metadata)
    metadata.setdefault("rubric_fingerprint", rubric.fingerprint)
    if rubric.id is not None:
        metadata.setdefault("rubric_id", rubric.id)
    if metadata == score.metadata:
        return score
    return Score(
        score.value,
        score.components,
        metadata,
        feedback=score.feedback,
    )


class GradingWrapper(
    gym.Wrapper[ObsT, ActT, ObsT, ActT],
    Generic[ObsT, ActT, InputT, RubricT],
):
    """Grade environment state inside synchronous ``step`` calls.

    ``make_input`` runs after the inner environment step, so it may inspect the
    action and the environment's resulting private state. Grading occurs on
    terminated or truncated steps by default.
    """

    def __init__(
        self,
        env: gym.Env[ObsT, ActT],
        *,
        rubric: RubricT,
        grader: Grader[InputT, RubricT],
        make_input: Callable[[gym.Env[ObsT, ActT], ActT], InputT],
        when: Callable[[bool, bool], bool] = _terminal,
        combine_reward: Callable[[float, Score], float] = _replace_reward,
    ) -> None:
        super().__init__(env)
        self.rubric = rubric
        self.grader = grader
        self.make_input = make_input
        self.when = when
        self.combine_reward = combine_reward

    def step(self, action: ActT) -> tuple[ObsT, float, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = self.env.step(action)
        scalar_reward = float(reward)
        if not self.when(terminated, truncated):
            return observation, scalar_reward, terminated, truncated, info
        value = self.grader(self.make_input(self.env, action), self.rubric)
        if inspect.isawaitable(value):
            close = getattr(value, "close", None)
            if callable(close):
                close()
            raise TypeError(
                "synchronous grading wrapper received an awaitable; "
                "use AsyncGradingWrapper"
            )
        score = _with_rubric_metadata(Score.from_value(value), self.rubric)
        resolved_info = dict(info)
        resolved_info.update(score.as_info())
        return (
            observation,
            float(self.combine_reward(scalar_reward, score)),
            terminated,
            truncated,
            resolved_info,
        )


class AsyncWrapper(
    AsyncEnv[WrapperObsT, WrapperActT],
    Generic[WrapperObsT, WrapperActT, ObsT, ActT],
):
    """Base wrapper that delegates to an inner async environment by default."""

    def __init__(self, env: AsyncEnv[ObsT, ActT]) -> None:
        self.env = env
        self.action_space: Space[WrapperActT] = cast(
            Space[WrapperActT], env.action_space
        )
        self.observation_space: Space[WrapperObsT] = cast(
            Space[WrapperObsT], env.observation_space
        )
        self.metadata = env.metadata

    @property
    def unwrapped(self) -> AsyncEnv[Any, Any]:
        env: AsyncEnv[Any, Any] = self.env
        while isinstance(env, AsyncWrapper):
            env = env.env
        return env

    async def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[WrapperObsT, dict[str, Any]]:
        observation, info = await self.env.reset(seed=seed, options=options)
        return cast(WrapperObsT, observation), info

    async def step(
        self, action: WrapperActT
    ) -> tuple[WrapperObsT, float, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = await self.env.step(
            cast(ActT, action)
        )
        return cast(WrapperObsT, observation), reward, terminated, truncated, info

    async def close(self) -> None:
        await self.env.close()


class AsyncActionWrapper(
    AsyncWrapper[ObsT, WrapperActT, ObsT, ActT],
    Generic[ObsT, WrapperActT, ActT],
):
    """Transform policy-facing actions into the inner environment's actions."""

    async def step(
        self, action: WrapperActT
    ) -> tuple[ObsT, float, bool, bool, dict[str, Any]]:
        return await self.env.step(await self.action(action))

    @abstractmethod
    async def action(self, action: WrapperActT) -> ActT:
        """Map an action from the wrapper space to the inner action space."""
        raise NotImplementedError


class AsyncObservationWrapper(
    AsyncWrapper[WrapperObsT, ActT, ObsT, ActT],
    Generic[WrapperObsT, ActT, ObsT],
):
    """Transform inner observations into policy-facing observations."""

    async def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[WrapperObsT, dict[str, Any]]:
        observation, info = await self.env.reset(seed=seed, options=options)
        return await self.observation(observation), info

    async def step(
        self, action: ActT
    ) -> tuple[WrapperObsT, float, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = await self.env.step(action)
        return (
            await self.observation(observation),
            reward,
            terminated,
            truncated,
            info,
        )

    @abstractmethod
    async def observation(self, observation: ObsT) -> WrapperObsT:
        """Map an inner observation to the wrapper observation space."""
        raise NotImplementedError


class AsyncRewardWrapper(AsyncWrapper[ObsT, ActT, ObsT, ActT], Generic[ObsT, ActT]):
    """Transform scalar rewards returned by an async environment."""

    async def step(
        self, action: ActT
    ) -> tuple[ObsT, float, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = await self.env.step(action)
        return (
            observation,
            float(await self.reward(reward)),
            terminated,
            truncated,
            info,
        )

    @abstractmethod
    async def reward(self, reward: float) -> float:
        """Map an inner scalar reward to the wrapper's scalar reward."""
        raise NotImplementedError


class AsyncGradingWrapper(
    AsyncWrapper[ObsT, ActT, ObsT, ActT],
    Generic[ObsT, ActT, InputT, RubricT],
):
    """Grade environment state inside asynchronous ``step`` calls."""

    def __init__(
        self,
        env: AsyncEnv[ObsT, ActT],
        *,
        rubric: RubricT,
        grader: Grader[InputT, RubricT],
        make_input: Callable[[AsyncEnv[ObsT, ActT], ActT], InputT],
        when: Callable[[bool, bool], bool] = _terminal,
        combine_reward: Callable[[float, Score], float] = _replace_reward,
    ) -> None:
        super().__init__(env)
        self.rubric = rubric
        self.grader = grader
        self.make_input = make_input
        self.when = when
        self.combine_reward = combine_reward

    async def step(
        self, action: ActT
    ) -> tuple[ObsT, float, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = await self.env.step(action)
        scalar_reward = float(reward)
        if not self.when(terminated, truncated):
            return observation, scalar_reward, terminated, truncated, info
        value = self.grader(self.make_input(self.env, action), self.rubric)
        if inspect.isawaitable(value):
            value = await value
        score = _with_rubric_metadata(Score.from_value(value), self.rubric)
        resolved_info = dict(info)
        resolved_info.update(score.as_info())
        return (
            observation,
            float(self.combine_reward(scalar_reward, score)),
            terminated,
            truncated,
            resolved_info,
        )
