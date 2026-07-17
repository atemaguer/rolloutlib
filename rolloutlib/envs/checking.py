"""Conformance checks for native asynchronous environments."""

from __future__ import annotations

from typing import Any

from .core import AsyncEnv


async def check_async_env(
    env: AsyncEnv[Any, Any],
    *,
    seed: int = 0,
    action: Any | None = None,
) -> None:
    """Run basic Gymnasium-style checks against an async environment.

    This intentionally mirrors the most important value-level checks from
    Gymnasium's checker without assuming that a random action is valid for
    every application-specific environment.
    Args:
        env: Asynchronous environment to validate.
        seed: Seed used for the reset check.
        action: Optional explicit action; otherwise one is sampled.

    Returns:
        ``None``. Raises ``AssertionError`` when a contract check fails.
    """
    observation, info = await env.reset(seed=seed)
    if not isinstance(info, dict):
        raise AssertionError("reset() info must be a dictionary")
    if observation not in env.observation_space:
        raise AssertionError("reset() observation is outside observation_space")

    selected_action = env.action_space.sample() if action is None else action
    if selected_action not in env.action_space:
        raise AssertionError("check action is outside action_space")
    next_observation, reward, terminated, truncated, step_info = await env.step(
        selected_action
    )
    if next_observation not in env.observation_space:
        raise AssertionError("step() observation is outside observation_space")
    if not isinstance(reward, (int, float)):
        raise AssertionError("step() reward must be numeric")
    if not isinstance(terminated, bool) or not isinstance(truncated, bool):
        raise AssertionError("step() termination flags must be bools")
    if not isinstance(step_info, dict):
        raise AssertionError("step() info must be a dictionary")


__all__ = ["check_async_env"]
