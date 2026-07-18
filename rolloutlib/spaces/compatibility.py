"""Compatibility checks for values flowing between Gymnasium spaces."""

from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium import Space
from gymnasium.spaces import (
    Box,
    Dict,
    Discrete,
    MultiBinary,
    MultiDiscrete,
    Sequence,
    Text,
    Tuple,
)

from ._pydantic import PydanticSpace
from .messages import ChatSpace, MessageSpace
from .text import TextSpace
from .tools import ToolCallSpace


def require_space(value: object, *, name: str) -> Space[Any]:
    """Return ``value`` as a Gymnasium space or raise a specific error."""

    if not isinstance(value, Space):
        raise TypeError(f"{name} must be a gymnasium.Space")
    return value


def _maximum_length(value: int | None) -> float:
    return float("inf") if value is None else value


def _dtype_subset(produced: Any, accepted: Any) -> bool:
    if produced is None or accepted is None:
        return produced is accepted
    return bool(np.can_cast(produced, accepted, casting="safe"))


def is_space_subset(produced: Space[Any], accepted: Space[Any]) -> bool:
    """Return whether every value in ``produced`` is accepted by ``accepted``.

    The check is structural for Gymnasium's standard container and scalar
    spaces plus rolloutlib's language spaces. Unknown custom spaces are treated
    conservatively and must compare equal.
    """

    if produced is accepted or produced == accepted:
        return True

    if isinstance(accepted, PydanticSpace) and accepted.annotation in (Any, object):
        return True

    if isinstance(produced, TextSpace) and isinstance(accepted, TextSpace):
        return (
            produced.min_length >= accepted.min_length
            and _maximum_length(produced.max_length)
            <= _maximum_length(accepted.max_length)
        )

    if isinstance(produced, MessageSpace) and isinstance(accepted, MessageSpace):
        return set(produced.roles).issubset(accepted.roles) and is_space_subset(
            produced.content_space,
            accepted.content_space,
        )

    if isinstance(produced, ChatSpace) and isinstance(accepted, ChatSpace):
        return (
            produced.min_length >= accepted.min_length
            and _maximum_length(produced.max_length)
            <= _maximum_length(accepted.max_length)
            and is_space_subset(produced.message_space, accepted.message_space)
        )

    if isinstance(produced, ToolCallSpace) and isinstance(accepted, ToolCallSpace):
        return set(produced.tools).issubset(accepted.tools) and all(
            is_space_subset(schema, accepted.tools[name])
            for name, schema in produced.tools.items()
        )

    if isinstance(produced, PydanticSpace) and isinstance(accepted, PydanticSpace):
        if produced.annotation == accepted.annotation:
            return True
        if isinstance(produced.annotation, type) and isinstance(
            accepted.annotation, type
        ):
            return issubclass(produced.annotation, accepted.annotation)
        return False

    if isinstance(produced, Discrete) and isinstance(accepted, Discrete):
        produced_end = int(produced.start + produced.n)
        accepted_end = int(accepted.start + accepted.n)
        return (
            int(produced.start) >= int(accepted.start)
            and produced_end <= accepted_end
            and _dtype_subset(produced.dtype, accepted.dtype)
        )

    if isinstance(produced, Box) and isinstance(accepted, Box):
        return (
            produced.shape == accepted.shape
            and _dtype_subset(produced.dtype, accepted.dtype)
            and bool(np.all(produced.low >= accepted.low))
            and bool(np.all(produced.high <= accepted.high))
        )

    if isinstance(produced, MultiDiscrete) and isinstance(accepted, MultiDiscrete):
        produced_high = produced.start + produced.nvec
        accepted_high = accepted.start + accepted.nvec
        return (
            produced.shape == accepted.shape
            and _dtype_subset(produced.dtype, accepted.dtype)
            and bool(np.all(produced.start >= accepted.start))
            and bool(np.all(produced_high <= accepted_high))
        )

    if isinstance(produced, MultiBinary) and isinstance(accepted, MultiBinary):
        return produced.shape == accepted.shape and _dtype_subset(
            produced.dtype, accepted.dtype
        )

    if isinstance(produced, Text) and isinstance(accepted, Text):
        return (
            produced.min_length >= accepted.min_length
            and produced.max_length <= accepted.max_length
            and set(produced.character_set).issubset(accepted.character_set)
        )

    if isinstance(produced, Dict) and isinstance(accepted, Dict):
        return set(produced.spaces) == set(accepted.spaces) and all(
            is_space_subset(space, accepted.spaces[name])
            for name, space in produced.spaces.items()
        )

    if isinstance(produced, Tuple) and isinstance(accepted, Tuple):
        return len(produced.spaces) == len(accepted.spaces) and all(
            is_space_subset(source, target)
            for source, target in zip(
                produced.spaces,
                accepted.spaces,
                strict=True,
            )
        )

    if isinstance(produced, Sequence) and isinstance(accepted, Sequence):
        return produced.stack == accepted.stack and is_space_subset(
            produced.feature_space,
            accepted.feature_space,
        )

    return False


def check_space_compatibility(
    produced: Space[Any],
    accepted: Space[Any],
    *,
    produced_name: str = "produced space",
    accepted_name: str = "accepted space",
) -> None:
    """Raise when values from one space cannot safely flow into another."""

    require_space(produced, name=produced_name)
    require_space(accepted, name=accepted_name)
    if not is_space_subset(produced, accepted):
        raise TypeError(
            f"{produced_name} {produced!r} is incompatible with "
            f"{accepted_name} {accepted!r}"
        )


def check_space_value(
    space: Space[Any],
    value: object,
    *,
    name: str,
) -> None:
    """Raise when ``value`` is outside a declared Gymnasium space."""

    require_space(space, name=f"{name} space")
    if value not in space:
        raise ValueError(f"{name} is outside its declared space {space!r}")


__all__ = [
    "check_space_compatibility",
    "check_space_value",
    "is_space_subset",
    "require_space",
]
