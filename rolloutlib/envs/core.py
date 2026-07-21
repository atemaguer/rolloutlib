"""Unified synchronous and asynchronous environment foundations."""

from __future__ import annotations

from abc import ABC, abstractmethod
import math
from typing import Any, TypeAlias, TypeVar

import gymnasium as gym

from .._awaitables import MaybeAwaitable, map_result
from ..graders import Score, ScoreValue
from ..spaces.compatibility import check_space_value, require_space


ObsT = TypeVar("ObsT")
ActT = TypeVar("ActT")

ResetResult: TypeAlias = tuple[ObsT, dict[str, Any]]
StepResult: TypeAlias = tuple[ObsT, float, bool, bool, dict[str, Any]]
EvaluationResult = ScoreValue | tuple[ScoreValue, dict[str, Any]]


def _resolve_evaluation(value: EvaluationResult) -> tuple[float, dict[str, Any]]:
    """Convert an evaluation result into a scalar reward and info mapping."""

    if isinstance(value, tuple):
        result, info = value
    else:
        result, info = value, {}
    if not isinstance(info, dict):
        raise TypeError("evaluation info must be a dictionary")
    resolved_info = dict(info)
    if isinstance(result, Score):
        resolved_info.update(result.as_info())
        return result.value, resolved_info
    if isinstance(result, bool):
        raise TypeError("evaluation score must be a finite number")
    reward = float(result)
    if not math.isfinite(reward):
        raise ValueError("evaluation score must be finite")
    return reward, resolved_info


class Env(gym.Env[ObsT, ActT], ABC):
    """An environment whose operations may return values or awaitables.

    ``reset``, ``step``, and ``close`` have the usual Gymnasium value-level
    contract. Implementations can define each operation with either ``def`` or
    ``async def``. Synchronous consumers require immediate values; async
    consumers await an operation only when it is awaitable.

    Async ``reset`` implementations should call ``super().reset(seed=seed)``
    without awaiting it to retain Gymnasium's RNG handling.
    """

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> Any:
        """Initialize Gymnasium RNG state for subclasses that call ``super``."""

        del options
        super().reset(seed=seed)
        return None

    @abstractmethod
    def step(self, action: ActT) -> Any:
        """Advance the environment by one action."""

        raise NotImplementedError

    def close(self) -> Any:
        """Release environment resources when the implementation owns any."""

        return None


class SingleTurnEnv(Env[ObsT, ActT], ABC):
    """A one-action environment with synchronous or asynchronous hooks."""

    def __init__(self) -> None:
        super().__init__()
        self._episode_active = False

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> MaybeAwaitable[ResetResult[ObsT]]:
        """Reset and obtain the initial observation from the implementation."""

        gym.Env.reset(self, seed=seed)
        require_space(self.action_space, name="environment action_space")
        require_space(self.observation_space, name="environment observation_space")

        def finish(value: ResetResult[ObsT]) -> ResetResult[ObsT]:
            observation, info = value
            check_space_value(
                self.observation_space,
                observation,
                name="initial observation",
            )
            if not isinstance(info, dict):
                raise TypeError("initial observation info must be a dictionary")
            self._episode_active = True
            return observation, info

        return map_result(self.initial_observation(options=options), finish)

    def step(self, action: ActT) -> MaybeAwaitable[StepResult[ObsT]]:
        """Evaluate one action and complete the episode."""

        self._require_active_episode()
        check_space_value(self.action_space, action, name="environment action")

        def after_evaluation(evaluation: EvaluationResult) -> MaybeAwaitable[StepResult[ObsT]]:
            reward, info = _resolve_evaluation(evaluation)

            def finish(observation: ObsT) -> StepResult[ObsT]:
                check_space_value(
                    self.observation_space,
                    observation,
                    name="terminal observation",
                )
                self._episode_active = False
                return observation, reward, True, False, info

            return map_result(self.terminal_observation(action), finish)

        return map_result(self.evaluate(action), after_evaluation)

    def _require_active_episode(self) -> None:
        if not self._episode_active:
            raise RuntimeError(
                "reset() must be called before step() and after an episode ends"
            )

    @abstractmethod
    def initial_observation(
        self, *, options: dict[str, Any] | None = None
    ) -> MaybeAwaitable[ResetResult[ObsT]]:
        """Return the initial observation and its info mapping."""

        raise NotImplementedError

    @abstractmethod
    def evaluate(self, action: ActT) -> MaybeAwaitable[EvaluationResult]:
        """Score an action, optionally with additional step information."""

        raise NotImplementedError

    @abstractmethod
    def terminal_observation(self, action: ActT) -> MaybeAwaitable[ObsT]:
        """Return the observation after the action has completed the episode."""

        raise NotImplementedError


__all__ = ["Env", "EvaluationResult", "ResetResult", "SingleTurnEnv", "StepResult"]
