"""Gymnasium-backed token spaces."""

from __future__ import annotations

from gymnasium.spaces import Discrete, Sequence


def id(
    vocab_size: int,
    *,
    start: int = 0,
    seed: int | None = None,
) -> Discrete:
    """Construct a space for a single token id.

    Args:
        vocab_size: Number of token IDs in the vocabulary.
        start: First integer represented by the space.
        seed: Optional random seed.

    Returns:
        Gymnasium ``Discrete`` space for token IDs.
    """

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
    """Construct a variable-length sequence space of token IDs.

    Args:
        vocab_size: Number of token IDs in the vocabulary.
        start: First integer represented by the element space.
        stack: Whether sampled sequences should be stacked when possible.
        seed: Optional random seed.

    Returns:
        Gymnasium ``Sequence`` space containing token-id elements.
    """

    return Sequence(
        id(vocab_size, start=start),
        seed=seed,
        stack=stack,
    )


__all__ = ["id", "sequence"]
