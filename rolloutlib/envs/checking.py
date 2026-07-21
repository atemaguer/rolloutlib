"""Conformance checks for unified environments."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .._awaitables import resolve
from ..spaces.compatibility import check_space_value, require_space
from .core import Env


async def check_env(
    env: Env[Any, Any],
    *,
    seed: int = 0,
    action: Any | None = None,
) -> None:
    """Run Gymnasium-style checks against sync or async implementations."""

    require_space(env.action_space, name="environment action_space")
    require_space(env.observation_space, name="environment observation_space")
    observation, info = await resolve(env.reset(seed=seed))
    if not isinstance(info, dict):
        raise AssertionError("reset() info must be a dictionary")
    try:
        check_space_value(env.observation_space, observation, name="reset observation")
    except ValueError as error:
        raise AssertionError(str(error)) from error

    selected_action = env.action_space.sample() if action is None else action
    try:
        check_space_value(env.action_space, selected_action, name="check action")
    except ValueError as error:
        raise AssertionError(str(error)) from error
    next_observation, reward, terminated, truncated, step_info = await resolve(
        env.step(selected_action)
    )
    try:
        check_space_value(env.observation_space, next_observation, name="step observation")
    except ValueError as error:
        raise AssertionError(str(error)) from error
    if isinstance(reward, bool):
        raise AssertionError("step() reward must be numeric")
    try:
        scalar_reward = float(reward)
    except (TypeError, ValueError) as error:
        raise AssertionError("step() reward must be numeric") from error
    if not math.isfinite(scalar_reward):
        raise AssertionError("step() reward must be finite")
    if not isinstance(terminated, (bool, np.bool_)) or not isinstance(
        truncated, (bool, np.bool_)
    ):
        raise AssertionError("step() termination flags must be bools")
    if not isinstance(step_info, dict):
        raise AssertionError("step() info must be a dictionary")


__all__ = ["check_env"]
