"""A unified wrapper that applies graders to selected environment steps."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Generic, TypeVar, cast

from gymnasium.spaces import Space

from .._awaitables import MaybeAwaitable, map_result
from ..envs.core import Env, StepResult
from ..graders import Grader, Score
from ..spaces.compatibility import (
    check_space_compatibility,
    check_space_value,
    require_space,
)
from .core import Wrapper


ObsT = TypeVar("ObsT")
ActT = TypeVar("ActT")
InputT = TypeVar("InputT")


def _terminal(terminated: bool, truncated: bool) -> bool:
    return terminated or truncated


def _replace_reward(reward: float, score: Score) -> float:
    del reward
    return score.value


class GradingWrapper(
    Wrapper[ObsT, ActT, ObsT, ActT],
    Generic[ObsT, ActT, InputT],
):
    """Apply a synchronous or asynchronous grader inside ``step``.

    The wrapper is immediate when its environment and grader are immediate;
    otherwise ``step`` returns an awaitable. This permits one composition to
    work with either rollout entry point.
    """

    def __init__(
        self,
        env: Env[ObsT, ActT],
        *,
        grader: Grader[InputT],
        make_input: Callable[[Env[ObsT, ActT], ActT], InputT] | None = None,
        input_space: Space[InputT] | None = None,
        when: Callable[[bool, bool], bool] = _terminal,
        combine_reward: Callable[[float, Score], float] = _replace_reward,
    ) -> None:
        super().__init__(env)
        if not isinstance(grader, Grader):
            raise TypeError("grader must be a Grader")
        if make_input is None:
            if input_space is not None:
                raise TypeError("input_space must be omitted when grading actions directly")
            resolved_input_space = cast(Space[InputT], env.action_space)
            produced_name = "environment action_space"

            def action_input(environment: Env[ObsT, ActT], action: ActT) -> InputT:
                del environment
                return cast(InputT, action)

            resolved_make_input = action_input
        else:
            if input_space is None:
                raise TypeError("input_space is required when make_input is provided")
            resolved_input_space = cast(
                Space[InputT],
                require_space(input_space, name="make_input input_space"),
            )
            produced_name = "make_input input_space"
            resolved_make_input = make_input
        try:
            grader_input_space = require_space(
                grader.input_space,
                name="grader input_space",
            )
        except AttributeError as error:
            raise TypeError("grader must define an input_space") from error
        check_space_compatibility(
            resolved_input_space,
            grader_input_space,
            produced_name=produced_name,
            accepted_name="grader input_space",
        )
        self.grader = grader
        self.make_input = resolved_make_input
        self.grading_input_space = resolved_input_space
        self.when = when
        self.combine_reward = combine_reward

    def step(self, action: ActT) -> MaybeAwaitable[StepResult[ObsT]]:
        """Advance the environment and grade the resulting step when selected."""

        return map_result(self.env.step(action), lambda result: self._grade_step(result, action))

    def _grade_step(
        self,
        result: StepResult[ObsT],
        action: ActT,
    ) -> MaybeAwaitable[StepResult[ObsT]]:
        observation, reward, terminated, truncated, info = result
        scalar_reward = float(reward)
        if not self.when(terminated, truncated):
            return observation, scalar_reward, terminated, truncated, info
        grader_input = self.make_input(self.env, action)
        check_space_value(
            self.grading_input_space,
            grader_input,
            name="make_input result",
        )
        return map_result(
            self.grader.grade(grader_input),
            lambda score: self._scored_step(
                observation,
                scalar_reward,
                terminated,
                truncated,
                info,
                score,
            ),
        )

    def _scored_step(
        self,
        observation: ObsT,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
        score: Score,
    ) -> StepResult[ObsT]:
        resolved_info = dict(info)
        resolved_info.update(score.as_info())
        return (
            observation,
            float(self.combine_reward(reward, score)),
            terminated,
            truncated,
            resolved_info,
        )


__all__ = ["GradingWrapper"]
