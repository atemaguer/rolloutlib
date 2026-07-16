"""Synchronous and asynchronous environment foundations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Space
from gymnasium.utils import seeding

from ..graders import Score, ScoreValue


ObsT = TypeVar("ObsT")
ActT = TypeVar("ActT")


EvaluationResult = ScoreValue | tuple[ScoreValue, dict[str, Any]]


def _resolve_evaluation(value: EvaluationResult) -> tuple[float, dict[str, Any]]:
    if isinstance(value, tuple):
        result, info = value
    else:
        result, info = value, {}
    resolved_info = dict(info)
    if isinstance(result, Score):
        resolved_info.update(result.as_info())
        return result.value, resolved_info
    return float(result), resolved_info

# A convenience alias, deliberately not a Rolloutlib subclass. Synchronous
# implementations therefore retain exact Gymnasium identity and compatibility.
Env = gym.Env


class AsyncEnv(Generic[ObsT, ActT], ABC):
    """Async counterpart to :class:`gymnasium.Env`.

    The value-level contract intentionally matches Gymnasium: environments expose
    action and observation spaces, ``reset`` returns ``(observation, info)``, and
    ``step`` returns the standard five-tuple. Only the calling convention differs.

    Native implementations should call ``await super().reset(seed=seed)`` at the
    beginning of ``reset`` to opt into Gymnasium-compatible RNG handling.
    """

    metadata: dict[str, Any] = {"render_modes": []}
    action_space: Space[ActT]
    observation_space: Space[ObsT]

    _np_random: np.random.Generator | None = None
    _np_random_seed: int | None = None

    @abstractmethod
    async def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsT, dict[str, Any]]:
        """Reset the environment and return its first observation and info."""
        if seed is not None:
            self._np_random, self._np_random_seed = seeding.np_random(seed)

    @abstractmethod
    async def step(
        self, action: ActT
    ) -> tuple[ObsT, float, bool, bool, dict[str, Any]]:
        """Advance the environment by one step."""
        raise NotImplementedError

    async def close(self) -> None:
        """Release resources owned by the environment."""

    @property
    def np_random(self) -> np.random.Generator:
        """Return the environment RNG, initializing it lazily when necessary."""
        if self._np_random is None:
            self._np_random, self._np_random_seed = seeding.np_random()
        return self._np_random

    @property
    def np_random_seed(self) -> int:
        """Return the seed associated with :attr:`np_random`."""
        if self._np_random_seed is None:
            self._np_random, self._np_random_seed = seeding.np_random()
        return self._np_random_seed


class SingleTurnEnv(gym.Env[ObsT, ActT], ABC):
    """Gymnasium environment whose first action completes the episode.

    Subclasses provide the initial and terminal observations plus an evaluator.
    The base class owns the reset/step lifecycle and standard Gymnasium results.
    """

    def __init__(self) -> None:
        super().__init__()
        self._episode_active = False

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsT, dict[str, Any]]:
        super().reset(seed=seed)
        observation, info = self.initial_observation(options=options)
        self._episode_active = True
        return observation, info

    def step(self, action: ActT) -> tuple[ObsT, float, bool, bool, dict[str, Any]]:
        self._require_active_episode()
        reward, info = _resolve_evaluation(self.evaluate(action))
        observation = self.terminal_observation(action)
        self._episode_active = False
        return observation, reward, True, False, info

    def _require_active_episode(self) -> None:
        if not self._episode_active:
            raise RuntimeError(
                "reset() must be called before step() and after an episode ends"
            )

    @abstractmethod
    def initial_observation(
        self, *, options: dict[str, Any] | None = None
    ) -> tuple[ObsT, dict[str, Any]]:
        """Return the observation and info presented by ``reset``."""
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, action: ActT) -> EvaluationResult:
        """Grade the action and return a score, optionally with extra info."""
        raise NotImplementedError

    @abstractmethod
    def terminal_observation(self, action: ActT) -> ObsT:
        """Return the terminal observation produced after ``action``."""
        raise NotImplementedError


class AsyncSingleTurnEnv(AsyncEnv[ObsT, ActT], ABC):
    """Async environment whose first action completes the episode."""

    def __init__(self) -> None:
        self._episode_active = False

    async def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsT, dict[str, Any]]:
        await super().reset(seed=seed, options=options)
        observation, info = await self.initial_observation(options=options)
        self._episode_active = True
        return observation, info

    async def step(
        self, action: ActT
    ) -> tuple[ObsT, float, bool, bool, dict[str, Any]]:
        self._require_active_episode()
        reward, info = _resolve_evaluation(await self.evaluate(action))
        observation = await self.terminal_observation(action)
        self._episode_active = False
        return observation, reward, True, False, info

    def _require_active_episode(self) -> None:
        if not self._episode_active:
            raise RuntimeError(
                "reset() must be called before step() and after an episode ends"
            )

    @abstractmethod
    async def initial_observation(
        self, *, options: dict[str, Any] | None = None
    ) -> tuple[ObsT, dict[str, Any]]:
        """Return the observation and info presented by ``reset``."""
        raise NotImplementedError

    @abstractmethod
    async def evaluate(self, action: ActT) -> EvaluationResult:
        """Grade the action and return a score, optionally with extra info."""
        raise NotImplementedError

    @abstractmethod
    async def terminal_observation(self, action: ActT) -> ObsT:
        """Return the terminal observation produced after ``action``."""
        raise NotImplementedError
