"""Spaces for ordinary Python strings."""

from __future__ import annotations

import string
from typing import Annotated

import numpy as np
from pydantic import StringConstraints

from ._pydantic import PydanticSpace


DEFAULT_ALPHABET = string.ascii_letters + string.digits + " .,!?-_\n"


class TextSpace(PydanticSpace[str]):
    """A length-constrained string space with deterministic seeded sampling."""

    def __init__(
        self,
        *,
        min_length: int = 0,
        max_length: int | None = None,
        sample_max_length: int = 32,
        sample_alphabet: str = DEFAULT_ALPHABET,
        seed: int | None = None,
    ) -> None:
        """Initialize a constrained text space.

        Args:
            min_length: Minimum accepted and sampled string length.
            max_length: Optional maximum accepted string length.
            sample_max_length: Maximum length used by the sampler.
            sample_alphabet: Characters available to the sampler.
            seed: Optional random seed.

        Returns:
            ``None``.
        """
        if min_length < 0:
            raise ValueError("min_length must be non-negative")
        if max_length is not None and max_length < min_length:
            raise ValueError("max_length must be at least min_length")
        if sample_max_length < min_length:
            raise ValueError("sample_max_length must be at least min_length")
        if not sample_alphabet:
            raise ValueError("sample_alphabet must not be empty")

        self.min_length = min_length
        self.max_length = max_length
        self.sample_max_length = (
            sample_max_length
            if max_length is None
            else min(sample_max_length, max_length)
        )
        self.sample_alphabet = sample_alphabet
        annotation = Annotated[
            str,
            StringConstraints(min_length=min_length, max_length=max_length),
        ]
        super().__init__(annotation, sampler=self._sample_text, seed=seed)

    def _sample_text(self, rng: np.random.Generator) -> str:
        """Sample a seeded random string from the configured alphabet.

        Args:
            rng: NumPy random generator supplied by the space.

        Returns:
            Random string within the configured length bounds.
        """
        length = int(rng.integers(self.min_length, self.sample_max_length + 1))
        indexes = rng.integers(0, len(self.sample_alphabet), size=length)
        return "".join(self.sample_alphabet[int(index)] for index in indexes)

    def __repr__(self) -> str:
        """Return a concise representation of the text space.

        Returns:
            Human-readable length constraints.
        """
        return f"TextSpace(min_length={self.min_length}, max_length={self.max_length})"


def text(
    *,
    min_length: int = 0,
    max_length: int | None = None,
    sample_max_length: int = 32,
    sample_alphabet: str = DEFAULT_ALPHABET,
    seed: int | None = None,
) -> TextSpace:
    """Construct a length-constrained text space.

    Args:
        min_length: Minimum accepted and sampled string length.
        max_length: Optional maximum accepted string length.
        sample_max_length: Maximum length used by the sampler.
        sample_alphabet: Characters available to the sampler.
        seed: Optional random seed.

    Returns:
        Configured ``TextSpace`` instance.
    """
    return TextSpace(
        min_length=min_length,
        max_length=max_length,
        sample_max_length=sample_max_length,
        sample_alphabet=sample_alphabet,
        seed=seed,
    )


__all__ = ["DEFAULT_ALPHABET", "TextSpace", "text"]
