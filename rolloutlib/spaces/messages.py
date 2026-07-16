"""Message and chat spaces."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import numpy as np
from gymnasium import Space

from rolloutlib.types import Chat, Message, Role

from ._pydantic import PydanticSpace
from .text import text


DEFAULT_ROLES: tuple[Role, ...] = ("system", "user", "assistant", "tool")


class MessageSpace(PydanticSpace[Message]):
    """A structural space for backend-neutral chat messages."""

    def __init__(
        self,
        *,
        roles: Sequence[Role] = DEFAULT_ROLES,
        content_space: Space[str] | None = None,
        seed: int | None = None,
    ) -> None:
        if not roles:
            raise ValueError("roles must not be empty")
        if any(not isinstance(role, str) or not role for role in roles):
            raise ValueError("roles must be non-empty strings")
        self.roles = tuple(roles)
        self.content_space = content_space or text()
        super().__init__(Message, sampler=self._sample_message, seed=seed)

    def _sample_message(self, rng: np.random.Generator) -> Message:
        role = self.roles[int(rng.integers(0, len(self.roles)))]
        return {"role": role, "content": self.content_space.sample()}

    def contains(self, x: object) -> bool:
        if not super().contains(x):
            return False
        assert isinstance(x, dict)
        if x["role"] not in self.roles:
            return False
        content = x["content"]
        return not isinstance(content, str) or content in self.content_space

    def seed(self, seed: int | None = None) -> list[int]:
        own_seed = cast(int, super().seed(seed))
        child_seed = int(self.np_random.integers(0, np.iinfo(np.int32).max))
        self.content_space.seed(child_seed)
        # Child-seed derivation should not advance this space's sampling stream.
        super().seed(own_seed)
        return [own_seed, child_seed]


class ChatSpace(PydanticSpace[Chat]):
    """A variable-length list of messages."""

    def __init__(
        self,
        message_space: MessageSpace | None = None,
        *,
        min_length: int = 0,
        max_length: int | None = None,
        sample_max_length: int = 8,
        seed: int | None = None,
    ) -> None:
        if min_length < 0:
            raise ValueError("min_length must be non-negative")
        if max_length is not None and max_length < min_length:
            raise ValueError("max_length must be at least min_length")
        if sample_max_length < min_length:
            raise ValueError("sample_max_length must be at least min_length")
        self.message_space = message_space or MessageSpace()
        self.min_length = min_length
        self.max_length = max_length
        self.sample_max_length = (
            sample_max_length
            if max_length is None
            else min(sample_max_length, max_length)
        )
        super().__init__(Chat, sampler=self._sample_chat, seed=seed)

    def _sample_chat(self, rng: np.random.Generator) -> Chat:
        length = int(rng.integers(self.min_length, self.sample_max_length + 1))
        return [self.message_space.sample() for _ in range(length)]

    def contains(self, x: object) -> bool:
        if not super().contains(x):
            return False
        assert isinstance(x, list)
        if len(x) < self.min_length:
            return False
        if self.max_length is not None and len(x) > self.max_length:
            return False
        return all(item in self.message_space for item in x)

    def seed(self, seed: int | None = None) -> list[int]:
        own_seed = cast(int, super().seed(seed))
        child_seed = int(self.np_random.integers(0, np.iinfo(np.int32).max))
        self.message_space.seed(child_seed)
        super().seed(own_seed)
        return [own_seed, child_seed]


def message(
    *,
    roles: Sequence[Role] = DEFAULT_ROLES,
    content_space: Space[str] | None = None,
    seed: int | None = None,
) -> MessageSpace:
    return MessageSpace(roles=roles, content_space=content_space, seed=seed)


def chat(
    message_space: MessageSpace | None = None,
    *,
    min_length: int = 0,
    max_length: int | None = None,
    sample_max_length: int = 8,
    seed: int | None = None,
) -> ChatSpace:
    return ChatSpace(
        message_space,
        min_length=min_length,
        max_length=max_length,
        sample_max_length=sample_max_length,
        seed=seed,
    )


__all__ = [
    "ChatSpace",
    "DEFAULT_ROLES",
    "MessageSpace",
    "chat",
    "message",
]
