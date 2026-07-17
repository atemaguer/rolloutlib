"""Gymnasium wrappers that apply rolloutlib graders to environment steps."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any, Generic, TypeVar

import gymnasium as gym

from ..envs.core import AsyncEnv
from ..graders import AsyncGrader, Grader, Score
from .async_ import AsyncWrapper


ObsT = TypeVar("ObsT")
ActT = TypeVar("ActT")
InputT = TypeVar("InputT")


def _terminal(terminated: bool, truncated: bool) -> bool:
    """Return whether a step ended for either terminal reason."""

    return terminated or truncated


def _replace_reward(reward: float, score: Score) -> float:
    """Replace an environment reward with a grading score."""

    del reward
    return score.value


class GradingWrapper(
    gym.Wrapper[ObsT, ActT, ObsT, ActT],
    Generic[ObsT, ActT, InputT],
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
        grader: Grader[InputT],
        make_input: Callable[[gym.Env[ObsT, ActT], ActT], InputT],
        when: Callable[[bool, bool], bool] = _terminal,
        combine_reward: Callable[[float, Score], float] = _replace_reward,
    ) -> None:
        """Create a synchronous wrapper that grades selected environment steps.

        Args:
            env: Environment whose steps should be graded.
            grader: Synchronous grader that evaluates the input.
            make_input: Builds grader input from the inner environment and action.
            when: Selects which steps receive a grade.
            combine_reward: Combines the original reward and score.

        Returns:
            ``None``.
        """
        super().__init__(env)
        self.grader = grader
        self.make_input = make_input
        self.when = when
        self.combine_reward = combine_reward

    def step(self, action: ActT) -> tuple[ObsT, float, bool, bool, dict[str, Any]]:
        """Advance the environment and grade the resulting step when selected."""

        observation, reward, terminated, truncated, info = self.env.step(action)
        scalar_reward = float(reward)
        if not self.when(terminated, truncated):
            return observation, scalar_reward, terminated, truncated, info
        score = self.grader.grade(self.make_input(self.env, action))
        resolved_info = dict(info)
        resolved_info.update(score.as_info())
        return (
            observation,
            float(self.combine_reward(scalar_reward, score)),
            terminated,
            truncated,
            resolved_info,
        )


class AsyncGradingWrapper(
    AsyncWrapper[ObsT, ActT, ObsT, ActT],
    Generic[ObsT, ActT, InputT],
):
    """Grade environment state inside asynchronous ``step`` calls."""

    def __init__(
        self,
        env: AsyncEnv[ObsT, ActT],
        *,
        grader: Grader[InputT] | AsyncGrader[InputT],
        make_input: Callable[[AsyncEnv[ObsT, ActT], ActT], InputT],
        when: Callable[[bool, bool], bool] = _terminal,
        combine_reward: Callable[[float, Score], float] = _replace_reward,
    ) -> None:
        """Create an asynchronous wrapper that grades selected steps.

        Args:
            env: Asynchronous environment whose steps should be graded.
            grader: Synchronous or asynchronous grader that evaluates the input.
            make_input: Builds grader input from the inner environment and action.
            when: Selects which steps receive a grade.
            combine_reward: Combines the original reward and score.

        Returns:
            ``None``.
        """
        super().__init__(env)
        self.grader = grader
        self.make_input = make_input
        self.when = when
        self.combine_reward = combine_reward

    async def step(
        self, action: ActT
    ) -> tuple[ObsT, float, bool, bool, dict[str, Any]]:
        """Advance the environment and asynchronously grade selected steps."""

        observation, reward, terminated, truncated, info = await self.env.step(action)
        scalar_reward = float(reward)
        if not self.when(terminated, truncated):
            return observation, scalar_reward, terminated, truncated, info
        value = self.grader.grade(self.make_input(self.env, action))
        if inspect.isawaitable(value):
            value = await value
        score = value
        resolved_info = dict(info)
        resolved_info.update(score.as_info())
        return (
            observation,
            float(self.combine_reward(scalar_reward, score)),
            terminated,
            truncated,
            resolved_info,
        )


__all__ = ["AsyncGradingWrapper", "GradingWrapper"]
