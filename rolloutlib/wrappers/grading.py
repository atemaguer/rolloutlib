"""Gymnasium wrappers that apply rolloutlib graders to environment steps."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any, Generic, TypeVar, cast

import gymnasium as gym
from gymnasium.spaces import Space

from ..envs.core import AsyncEnv
from ..graders import AsyncGrader, Grader, Score
from ..spaces.compatibility import (
    check_space_compatibility,
    check_space_value,
    require_space,
)
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


def _grader_input_space(
    grader: Grader[Any] | AsyncGrader[Any],
) -> Space[Any]:
    try:
        value = grader.input_space
    except AttributeError as error:
        raise TypeError("grader must define an input_space") from error
    return require_space(value, name="grader input_space")


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
        make_input: Callable[[gym.Env[ObsT, ActT], ActT], InputT] | None = None,
        input_space: Space[InputT] | None = None,
        when: Callable[[bool, bool], bool] = _terminal,
        combine_reward: Callable[[float, Score], float] = _replace_reward,
    ) -> None:
        """Create a synchronous wrapper that grades selected environment steps.

        Args:
            env: Environment whose steps should be graded.
            grader: Synchronous grader that evaluates the input.
            make_input: Builds grader input from the inner environment and action.
                When omitted, the environment action is graded directly.
            input_space: Declared output space of ``make_input``. Required when
                ``make_input`` is provided.
            when: Selects which steps receive a grade.
            combine_reward: Combines the original reward and score.

        Returns:
            ``None``.
        """
        super().__init__(env)
        if not isinstance(grader, Grader):
            raise TypeError("GradingWrapper requires a synchronous Grader")
        if make_input is None:
            if input_space is not None:
                raise TypeError(
                    "input_space must be omitted when grading actions directly"
                )
            resolved_input_space = cast(Space[InputT], env.action_space)
            produced_name = "environment action_space"

            def action_input(
                environment: gym.Env[ObsT, ActT],
                action: ActT,
            ) -> InputT:
                del environment
                return cast(InputT, action)

            resolved_make_input = action_input
        else:
            if input_space is None:
                raise TypeError(
                    "input_space is required when make_input is provided"
                )
            resolved_input_space = cast(
                Space[InputT],
                require_space(input_space, name="make_input input_space"),
            )
            produced_name = "make_input input_space"
            resolved_make_input = make_input
        check_space_compatibility(
            resolved_input_space,
            _grader_input_space(grader),
            produced_name=produced_name,
            accepted_name="grader input_space",
        )
        self.grader = grader
        self.make_input = resolved_make_input
        self.grading_input_space = resolved_input_space
        self.when = when
        self.combine_reward = combine_reward

    def step(self, action: ActT) -> tuple[ObsT, float, bool, bool, dict[str, Any]]:
        """Advance the environment and grade the resulting step when selected."""

        observation, reward, terminated, truncated, info = self.env.step(action)
        scalar_reward = float(reward)
        if not self.when(terminated, truncated):
            return observation, scalar_reward, terminated, truncated, info
        grader_input = self.make_input(self.env, action)
        check_space_value(
            self.grading_input_space,
            grader_input,
            name="make_input result",
        )
        score = self.grader.grade(grader_input)
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
        make_input: Callable[[AsyncEnv[ObsT, ActT], ActT], InputT] | None = None,
        input_space: Space[InputT] | None = None,
        when: Callable[[bool, bool], bool] = _terminal,
        combine_reward: Callable[[float, Score], float] = _replace_reward,
    ) -> None:
        """Create an asynchronous wrapper that grades selected steps.

        Args:
            env: Asynchronous environment whose steps should be graded.
            grader: Synchronous or asynchronous grader that evaluates the input.
            make_input: Builds grader input from the inner environment and action.
                When omitted, the environment action is graded directly.
            input_space: Declared output space of ``make_input``. Required when
                ``make_input`` is provided.
            when: Selects which steps receive a grade.
            combine_reward: Combines the original reward and score.

        Returns:
            ``None``.
        """
        super().__init__(env)
        if not isinstance(grader, (Grader, AsyncGrader)):
            raise TypeError("grader must be a Grader or AsyncGrader")
        if make_input is None:
            if input_space is not None:
                raise TypeError(
                    "input_space must be omitted when grading actions directly"
                )
            resolved_input_space = cast(Space[InputT], env.action_space)
            produced_name = "environment action_space"

            def action_input(
                environment: AsyncEnv[ObsT, ActT],
                action: ActT,
            ) -> InputT:
                del environment
                return cast(InputT, action)

            resolved_make_input = action_input
        else:
            if input_space is None:
                raise TypeError(
                    "input_space is required when make_input is provided"
                )
            resolved_input_space = cast(
                Space[InputT],
                require_space(input_space, name="make_input input_space"),
            )
            produced_name = "make_input input_space"
            resolved_make_input = make_input
        check_space_compatibility(
            resolved_input_space,
            _grader_input_space(grader),
            produced_name=produced_name,
            accepted_name="grader input_space",
        )
        self.grader = grader
        self.make_input = resolved_make_input
        self.grading_input_space = resolved_input_space
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
        grader_input = self.make_input(self.env, action)
        check_space_value(
            self.grading_input_space,
            grader_input,
            name="make_input result",
        )
        value = self.grader.grade(grader_input)
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
