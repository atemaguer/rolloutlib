"""Composable wrappers for asynchronous environments."""

from __future__ import annotations

from abc import abstractmethod
import math
from typing import Any, Generic, TypeVar, cast

from gymnasium.spaces import Space

from ..envs.core import AsyncEnv
from ..spaces.compatibility import check_space_value, require_space


ObsT = TypeVar("ObsT")
ActT = TypeVar("ActT")
WrapperObsT = TypeVar("WrapperObsT")
WrapperActT = TypeVar("WrapperActT")


class AsyncWrapper(
    AsyncEnv[WrapperObsT, WrapperActT],
    Generic[WrapperObsT, WrapperActT, ObsT, ActT],
):
    """Base wrapper that delegates to an inner async environment by default."""

    def __init__(self, env: AsyncEnv[ObsT, ActT]) -> None:
        """Create a wrapper around an asynchronous environment.

        Args:
            env: Environment to delegate to.

        Returns:
            ``None``.
        """
        self.env = env
        self.action_space: Space[WrapperActT] = cast(
            Space[WrapperActT],
            require_space(env.action_space, name="environment action_space"),
        )
        self.observation_space: Space[WrapperObsT] = cast(
            Space[WrapperObsT],
            require_space(
                env.observation_space,
                name="environment observation_space",
            ),
        )
        self.metadata = env.metadata

    @property
    def unwrapped(self) -> AsyncEnv[Any, Any]:
        """Return the innermost asynchronous environment.

        Returns:
            The environment beneath all nested ``AsyncWrapper`` instances.
        """
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
        """Reset the wrapped environment.

        Args:
            seed: Optional reset seed.
            options: Optional environment-specific reset options.

        Returns:
            The wrapped initial observation and reset information.
        """
        observation, info = await self.env.reset(seed=seed, options=options)
        return cast(WrapperObsT, observation), info

    async def step(
        self, action: WrapperActT
    ) -> tuple[WrapperObsT, float, bool, bool, dict[str, Any]]:
        """Forward an action to the wrapped environment.

        Args:
            action: Action in the wrapper's action space.

        Returns:
            The wrapped environment's standard five-tuple.
        """
        observation, reward, terminated, truncated, info = await self.env.step(
            cast(ActT, action)
        )
        return cast(WrapperObsT, observation), reward, terminated, truncated, info

    async def close(self) -> None:
        """Close the wrapped environment.

        Returns:
            ``None``.
        """
        await self.env.close()


class AsyncActionWrapper(
    AsyncWrapper[ObsT, WrapperActT, ObsT, ActT],
    Generic[ObsT, WrapperActT, ActT],
):
    """Transform policy-facing actions into the inner environment's actions."""

    async def step(
        self, action: WrapperActT
    ) -> tuple[ObsT, float, bool, bool, dict[str, Any]]:
        """Transform an action and advance the inner environment.

        Args:
            action: Action in the wrapper's action space.

        Returns:
            The inner environment's standard five-tuple.
        """
        check_space_value(self.action_space, action, name="wrapper action")
        inner_action = await self.action(action)
        check_space_value(
            self.env.action_space,
            inner_action,
            name="transformed action",
        )
        return await self.env.step(inner_action)

    @abstractmethod
    async def action(self, action: WrapperActT) -> ActT:
        """Map an action from the wrapper space to the inner action space.

        Args:
            action: Action in the wrapper's action space.

        Returns:
            The transformed action accepted by the inner environment.
        """
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
        """Reset the inner environment and transform its observation.

        Args:
            seed: Optional reset seed.
            options: Optional environment-specific reset options.

        Returns:
            The transformed initial observation and reset information.
        """
        observation, info = await self.env.reset(seed=seed, options=options)
        check_space_value(
            self.env.observation_space,
            observation,
            name="inner reset observation",
        )
        transformed = await self.observation(observation)
        check_space_value(
            self.observation_space,
            transformed,
            name="transformed reset observation",
        )
        return transformed, info

    async def step(
        self, action: ActT
    ) -> tuple[WrapperObsT, float, bool, bool, dict[str, Any]]:
        """Advance the inner environment and transform its observation.

        Args:
            action: Action accepted by the inner environment.

        Returns:
            A five-tuple containing the transformed observation and step data.
        """
        observation, reward, terminated, truncated, info = await self.env.step(action)
        check_space_value(
            self.env.observation_space,
            observation,
            name="inner step observation",
        )
        transformed = await self.observation(observation)
        check_space_value(
            self.observation_space,
            transformed,
            name="transformed step observation",
        )
        return (
            transformed,
            reward,
            terminated,
            truncated,
            info,
        )

    @abstractmethod
    async def observation(self, observation: ObsT) -> WrapperObsT:
        """Map an inner observation to the wrapper observation space.

        Args:
            observation: Observation returned by the inner environment.

        Returns:
            The transformed observation exposed by the wrapper.
        """
        raise NotImplementedError


class AsyncRewardWrapper(AsyncWrapper[ObsT, ActT, ObsT, ActT], Generic[ObsT, ActT]):
    """Transform scalar rewards returned by an async environment."""

    async def step(
        self, action: ActT
    ) -> tuple[ObsT, float, bool, bool, dict[str, Any]]:
        """Advance the inner environment and transform its reward.

        Args:
            action: Action accepted by the inner environment.

        Returns:
            A five-tuple containing the transformed scalar reward.
        """
        observation, reward, terminated, truncated, info = await self.env.step(action)
        transformed = float(await self.reward(reward))
        if not math.isfinite(transformed):
            raise ValueError("transformed reward must be finite")
        return (
            observation,
            transformed,
            terminated,
            truncated,
            info,
        )

    @abstractmethod
    async def reward(self, reward: float) -> float:
        """Map an inner scalar reward to the wrapper's scalar reward.

        Args:
            reward: Scalar reward returned by the inner environment.

        Returns:
            The transformed scalar reward.
        """
        raise NotImplementedError


__all__ = [
    "AsyncActionWrapper",
    "AsyncObservationWrapper",
    "AsyncRewardWrapper",
    "AsyncWrapper",
]
