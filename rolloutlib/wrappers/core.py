"""Composable wrappers for unified environments."""

from __future__ import annotations

from abc import abstractmethod
import math
from typing import Any, Generic, TypeVar, cast

from gymnasium.spaces import Space

from .._awaitables import MaybeAwaitable, map_result
from ..envs.core import Env, ResetResult, StepResult
from ..spaces.compatibility import check_space_value, require_space


ObsT = TypeVar("ObsT")
ActT = TypeVar("ActT")
WrapperObsT = TypeVar("WrapperObsT")
WrapperActT = TypeVar("WrapperActT")


class Wrapper(
    Env[WrapperObsT, WrapperActT],
    Generic[WrapperObsT, WrapperActT, ObsT, ActT],
):
    """Delegate to an inner environment while preserving its calling style."""

    def __init__(self, env: Env[ObsT, ActT]) -> None:
        self.env = env
        self.action_space: Space[WrapperActT] = cast(
            Space[WrapperActT],
            require_space(env.action_space, name="environment action_space"),
        )
        self.observation_space: Space[WrapperObsT] = cast(
            Space[WrapperObsT],
            require_space(env.observation_space, name="environment observation_space"),
        )
        self.metadata = env.metadata

    @property
    def unwrapped(self) -> Env[Any, Any]:
        """Return the environment beneath all nested rolloutlib wrappers."""

        env: Env[Any, Any] = self.env
        while isinstance(env, Wrapper):
            env = env.env
        return env

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> MaybeAwaitable[ResetResult[WrapperObsT]]:
        """Reset the inner environment."""

        return cast(
            MaybeAwaitable[ResetResult[WrapperObsT]],
            self.env.reset(seed=seed, options=options),
        )

    def step(self, action: WrapperActT) -> MaybeAwaitable[StepResult[WrapperObsT]]:
        """Forward an action to the inner environment."""

        return cast(MaybeAwaitable[StepResult[WrapperObsT]], self.env.step(cast(ActT, action)))

    def close(self) -> MaybeAwaitable[None]:
        """Close the inner environment."""

        return self.env.close()


class ActionWrapper(
    Wrapper[ObsT, WrapperActT, ObsT, ActT],
    Generic[ObsT, WrapperActT, ActT],
):
    """Transform policy-facing actions into inner-environment actions."""

    def step(self, action: WrapperActT) -> MaybeAwaitable[StepResult[ObsT]]:
        check_space_value(self.action_space, action, name="wrapper action")

        def advance(inner_action: ActT) -> MaybeAwaitable[StepResult[ObsT]]:
            check_space_value(
                self.env.action_space,
                inner_action,
                name="transformed action",
            )
            return self.env.step(inner_action)

        return map_result(self.action(action), advance)

    @abstractmethod
    def action(self, action: WrapperActT) -> MaybeAwaitable[ActT]:
        """Map an action from the wrapper space into the inner action space."""

        raise NotImplementedError


class ObservationWrapper(
    Wrapper[WrapperObsT, ActT, ObsT, ActT],
    Generic[WrapperObsT, ActT, ObsT],
):
    """Transform inner observations into policy-facing observations."""

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> MaybeAwaitable[ResetResult[WrapperObsT]]:
        def transform(value: ResetResult[ObsT]) -> MaybeAwaitable[ResetResult[WrapperObsT]]:
            observation, info = value
            return map_result(self.observation(observation), lambda transformed: self._reset_result(transformed, info))

        return map_result(self.env.reset(seed=seed, options=options), transform)

    def _reset_result(
        self,
        observation: WrapperObsT,
        info: dict[str, Any],
    ) -> ResetResult[WrapperObsT]:
        check_space_value(self.observation_space, observation, name="transformed reset observation")
        return observation, info

    def step(self, action: ActT) -> MaybeAwaitable[StepResult[WrapperObsT]]:
        def transform(value: StepResult[ObsT]) -> MaybeAwaitable[StepResult[WrapperObsT]]:
            observation, reward, terminated, truncated, info = value
            return map_result(
                self.observation(observation),
                lambda transformed: self._step_result(transformed, reward, terminated, truncated, info),
            )

        return map_result(self.env.step(action), transform)

    def _step_result(
        self,
        observation: WrapperObsT,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
    ) -> StepResult[WrapperObsT]:
        check_space_value(self.observation_space, observation, name="transformed step observation")
        return observation, reward, terminated, truncated, info

    @abstractmethod
    def observation(self, observation: ObsT) -> MaybeAwaitable[WrapperObsT]:
        """Map an inner observation to the wrapper observation space."""

        raise NotImplementedError


class RewardWrapper(Wrapper[ObsT, ActT, ObsT, ActT], Generic[ObsT, ActT]):
    """Transform scalar rewards while preserving synchronous fast paths."""

    def step(self, action: ActT) -> MaybeAwaitable[StepResult[ObsT]]:
        def transform(value: StepResult[ObsT]) -> MaybeAwaitable[StepResult[ObsT]]:
            observation, reward, terminated, truncated, info = value
            return map_result(
                self.reward(reward),
                lambda transformed: self._step_result(
                    observation, transformed, terminated, truncated, info
                ),
            )

        return map_result(self.env.step(action), transform)

    def _step_result(
        self,
        observation: ObsT,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
    ) -> StepResult[ObsT]:
        transformed = float(reward)
        if not math.isfinite(transformed):
            raise ValueError("transformed reward must be finite")
        return observation, transformed, terminated, truncated, info

    @abstractmethod
    def reward(self, reward: float) -> MaybeAwaitable[float]:
        """Map an inner reward to the wrapper reward."""

        raise NotImplementedError


__all__ = ["ActionWrapper", "ObservationWrapper", "RewardWrapper", "Wrapper"]
