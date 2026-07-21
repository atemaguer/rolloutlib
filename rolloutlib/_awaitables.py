"""Utilities for APIs whose collaborators may be synchronous or awaitable."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from typing import TypeAlias, TypeVar


ValueT = TypeVar("ValueT")
ResultT = TypeVar("ResultT")

MaybeAwaitable: TypeAlias = ValueT | Awaitable[ValueT]


def is_awaitable(value: object) -> bool:
    """Return whether ``value`` must be awaited to obtain its result."""

    return inspect.isawaitable(value)


async def resolve(value: MaybeAwaitable[ValueT]) -> ValueT:
    """Return an immediate value or await it when necessary."""

    if inspect.isawaitable(value):
        return await value
    return value


def map_result(
    value: MaybeAwaitable[ValueT],
    transform: Callable[[ValueT], MaybeAwaitable[ResultT]],
) -> MaybeAwaitable[ResultT]:
    """Transform a value while preserving a synchronous fast path.

    The returned value is immediate exactly when both the input and transform
    are immediate. Otherwise it is a coroutine resolving the complete chain.
    """

    if not inspect.isawaitable(value):
        return transform(value)

    async def mapped() -> ResultT:
        return await resolve(transform(await value))

    return mapped()


def require_sync(value: MaybeAwaitable[ValueT], *, source: str) -> ValueT:
    """Return an immediate value or explain that the async entry point is needed."""

    if inspect.isawaitable(value):
        close = getattr(value, "close", None)
        if callable(close):
            close()
        raise TypeError(f"{source} returned an awaitable; use the async entry point")
    return value
