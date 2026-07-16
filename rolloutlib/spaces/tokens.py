"""Gymnasium-backed token spaces."""

from __future__ import annotations

from gymnasium.spaces import Discrete, Sequence


def id(
    vocab_size: int,
    *,
    start: int = 0,
    seed: int | None = None,
) -> Discrete:
    """Return a space for a single token id."""

    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")
    return Discrete(vocab_size, start=start, seed=seed)


def sequence(
    vocab_size: int,
    *,
    start: int = 0,
    stack: bool = False,
    seed: int | None = None,
) -> Sequence:
    """Return a variable-length sequence of token ids."""

    return Sequence(
        id(vocab_size, start=start),
        seed=seed,
        stack=stack,
    )


__all__ = ["id", "sequence"]
