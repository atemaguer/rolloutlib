"""Pydantic-backed Gymnasium spaces."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Generic, TypeVar

import numpy as np
from gymnasium import Space
from pydantic import TypeAdapter, ValidationError


T = TypeVar("T")
Sampler = Callable[[np.random.Generator], T]


class PydanticSpace(Space[T], Generic[T]):
    """A Gymnasium space whose membership is defined by a Pydantic type.

    ``annotation`` may be a ``TypedDict``, a standard Python type, or any other
    annotation supported by :class:`pydantic.TypeAdapter`. Validation is strict:
    membership never coerces the supplied value. A sampler is optional for the
    generic class, but domain spaces in rolloutlib provide one.
    """

    def __init__(
        self,
        annotation: Any,
        *,
        sampler: Sampler[T] | None = None,
        seed: int | None = None,
    ) -> None:
        """Initialize a Pydantic-validated Gymnasium space.

        Args:
            annotation: Pydantic-compatible type annotation for membership.
            sampler: Optional callable producing values from a NumPy generator.
            seed: Optional random seed for sampling.

        Returns:
            ``None``.
        """
        self.annotation = annotation
        self.adapter: TypeAdapter[T] = TypeAdapter(annotation)
        self._sampler = sampler
        super().__init__(shape=None, dtype=None, seed=seed)

    def validate(self, value: object) -> T:
        """Validate a value without coercing its type.

        Args:
            value: Candidate value to validate.

        Returns:
            Validated value with type ``T``.
        """

        return self.adapter.validate_python(value, strict=True)

    def contains(self, x: object) -> bool:
        """Check whether a value belongs to this space.

        Args:
            x: Candidate value.

        Returns:
            ``True`` when strict Pydantic validation succeeds.
        """
        try:
            self.validate(x)
        except (ValidationError, TypeError, ValueError):
            return False
        return True

    def sample(
        self,
        mask: Any | None = None,
        probability: Any | None = None,
    ) -> T:
        """Sample a value from the configured sampler.

        Args:
            mask: Unsupported Gymnasium sampling mask.
            probability: Unsupported Gymnasium sampling probability.

        Returns:
            Sampled value with type ``T``.
        """
        if mask is not None or probability is not None:
            raise NotImplementedError("PydanticSpace does not support sampling masks")
        if self._sampler is None:
            raise NotImplementedError("No sampler was supplied for this PydanticSpace")
        return self.validate(self._sampler(self.np_random))

    def to_jsonable(self, sample_n: Sequence[T]) -> list[Any]:
        """Serialize samples into JSON-compatible Python values.

        Args:
            sample_n: Sequence of values to serialize.

        Returns:
            JSON-compatible serialized values.
        """
        return [self.adapter.dump_python(value, mode="json") for value in sample_n]

    def from_jsonable(self, sample_n: Sequence[Any]) -> list[T]:
        """Deserialize and validate JSON-compatible samples.

        Args:
            sample_n: Sequence of serialized values.

        Returns:
            Validated values with type ``T``.
        """
        return [self.validate(value) for value in sample_n]

    def __repr__(self) -> str:
        """Return a concise representation of the space.

        Returns:
            Human-readable representation containing the annotation name.
        """
        name = getattr(self.annotation, "__name__", repr(self.annotation))
        return f"PydanticSpace({name})"


__all__ = ["PydanticSpace"]
