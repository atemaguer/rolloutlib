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
        """Normalize the dataset contents and metadata into immutable values.

        Returns:
            ``None``. The dataclass fields are normalized in place during
            initialization.
        """
        object.__setattr__(self, "items", tuple(self.items))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def __len__(self) -> int:
        """Return the number of items in the dataset.

        Returns:
            The dataset length as an integer.
        """
        return len(self.items)

    @overload
    def __getitem__(self, index: int) -> ItemT:
        """Return an item selected by an integer index."""
        ...

    @overload
    def __getitem__(self, index: slice) -> tuple[ItemT, ...]:
        """Return a tuple of items selected by a slice."""
        ...

    def __getitem__(self, index: int | slice) -> ItemT | tuple[ItemT, ...]:
        """Return one item or a tuple of items selected by ``index``.

        Args:
            index: Integer position or slice selecting dataset items.

        Returns:
            The selected item for an integer index, or a tuple for a slice.
        """
        return cast(ItemT | tuple[ItemT, ...], self.items[index])

    def __iter__(self) -> Iterator[ItemT]:
        """Return an iterator over dataset items.

        Returns:
            An iterator yielding items in dataset order.
        """
        return iter(self.items)

    def item_id(self, index: int) -> str:
        """Return a stable default identifier for an item.

        Args:
            index: Integer position of the item in the dataset.

        Returns:
            The index converted to a string.
        """

        return str(index)

    def map(self, function: Callable[[ItemT], OutputT]) -> "Dataset[OutputT]":
        """Apply ``function`` to every item and return a new dataset.

        Args:
            function: Callable that transforms one input item into an output
                item.

        Returns:
            A dataset containing the transformed items and copied metadata.
        """
        return Dataset(tuple(function(item) for item in self), metadata=self.metadata)

    def filter(self, predicate: Callable[[ItemT], bool]) -> "Dataset[ItemT]":
        """Return a new dataset containing items accepted by ``predicate``.

        Args:
            predicate: Callable returning ``True`` for items to retain.

        Returns:
            A dataset containing only accepted items and copied metadata.
        """
        return Dataset(
            tuple(item for item in self if predicate(item)),
            metadata=self.metadata,
        )

    def take(self, limit: int | None) -> "Dataset[ItemT]":
        """Return at most ``limit`` items from the beginning of the dataset.

        Args:
            limit: Maximum number of items, or ``None`` to copy all items.

        Returns:
            A dataset containing the selected prefix.

        Raises:
            ValueError: If ``limit`` is negative.
        """
        if limit is not None and limit < 0:
            raise ValueError("limit must be non-negative")
        items = self if limit is None else islice(self, limit)
        return Dataset(tuple(items), metadata=self.metadata)

    def with_metadata(self, **metadata: Any) -> "Dataset[ItemT]":
        """Return a copy with additional or replacement metadata.

        Args:
            **metadata: Metadata fields to merge over the existing values.

        Returns:
            A dataset with the same items and merged metadata.
        """
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
        """Return the configured identifier for the item at ``index``.

        Args:
            index: Integer position of the item in the dataset.

        Returns:
            The result of ``get_item_id`` when configured, otherwise the
            default string form of the index.
        """
        item = self[index]
        return self.get_item_id(item) if self.get_item_id else super().item_id(index)


__all__ = ["Dataset", "RLDataset"]
