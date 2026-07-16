"""Data-source abstractions for rollout and evaluation workloads."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from itertools import islice
from typing import Any, Generic, TypeVar, cast, overload


ItemT = TypeVar("ItemT")
EnvT = TypeVar("EnvT")
OutputT = TypeVar("OutputT")


@dataclass(frozen=True, slots=True)
class Dataset(Generic[ItemT], Sequence[ItemT]):
    """An immutable, ordered source of items.

    This concrete container is a convenience for small in-memory datasets.
    Larger or streaming datasets can implement the same sequence/iterable
    protocol without inheriting from this class.
    """

    items: Sequence[ItemT] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "items", tuple(self.items))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def __len__(self) -> int:
        return len(self.items)

    @overload
    def __getitem__(self, index: int) -> ItemT: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[ItemT, ...]: ...

    def __getitem__(self, index: int | slice) -> ItemT | tuple[ItemT, ...]:
        return cast(ItemT | tuple[ItemT, ...], self.items[index])

    def __iter__(self) -> Iterator[ItemT]:
        return iter(self.items)

    def item_id(self, index: int) -> str:
        """Return a stable default identifier for an item."""

        return str(index)

    def map(self, function: Callable[[ItemT], OutputT]) -> "Dataset[OutputT]":
        return Dataset(tuple(function(item) for item in self), metadata=self.metadata)

    def filter(self, predicate: Callable[[ItemT], bool]) -> "Dataset[ItemT]":
        return Dataset(
            tuple(item for item in self if predicate(item)),
            metadata=self.metadata,
        )

    def take(self, limit: int | None) -> "Dataset[ItemT]":
        if limit is not None and limit < 0:
            raise ValueError("limit must be non-negative")
        items = self if limit is None else islice(self, limit)
        return Dataset(tuple(items), metadata=self.metadata)

    def with_metadata(self, **metadata: Any) -> "Dataset[ItemT]":
        merged = dict(self.metadata)
        merged.update(metadata)
        return Dataset(self.items, metadata=merged)


@dataclass(frozen=True, slots=True)
class RLDataset(Dataset[ItemT], Generic[ItemT, EnvT]):
    """A dataset paired with a factory for fresh environment instances.

    Items are pre-rollout work, not completed trajectories. Sampling strategy,
    batch size, and the number of rollouts per item remain training-loop
    concerns.
    """

    make_env: Callable[[ItemT], EnvT] = field(kw_only=True)
    get_item_id: Callable[[ItemT], str] | None = field(default=None, kw_only=True)

    def item_id(self, index: int) -> str:
        item = self[index]
        return self.get_item_id(item) if self.get_item_id else super().item_id(index)


__all__ = ["Dataset", "RLDataset"]
