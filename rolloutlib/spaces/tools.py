"""Spaces for structured tool calls."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

import numpy as np
from gymnasium import Space
from gymnasium.spaces import Dict as DictSpace
from gymnasium.spaces import Sequence as SequenceSpace

from rolloutlib.types import ToolCall

from ._pydantic import PydanticSpace


class ToolCallSpace(PydanticSpace[ToolCall]):
    """Validate a tool name and its arguments with a tool-specific Gym space."""

    def __init__(
        self,
        tools: Mapping[str, Space[Any]],
        *,
        include_id_in_samples: bool = False,
        seed: int | None = None,
    ) -> None:
        if not tools:
            raise ValueError("at least one tool schema is required")
        if any(not isinstance(name, str) or not name for name in tools):
            raise ValueError("tool names must be non-empty strings")
        if any(not isinstance(space, Space) for space in tools.values()):
            raise TypeError("each tool schema must be a gymnasium Space")
        self.tools = dict(tools)
        self.include_id_in_samples = include_id_in_samples
        super().__init__(ToolCall, sampler=self._sample_call, seed=seed)

    @property
    def tool_schemas(self) -> Mapping[str, Space[Any]]:
        return self.tools

    def _sample_call(self, rng: np.random.Generator) -> ToolCall:
        names = tuple(self.tools)
        name = names[int(rng.integers(0, len(names)))]
        value: ToolCall = {
            "name": name,
            "arguments": self.tools[name].sample(),
        }
        if self.include_id_in_samples:
            value["id"] = f"call_{int(rng.integers(0, 2**63)):016x}"
        return value

    def contains(self, x: object) -> bool:
        if not super().contains(x):
            return False
        assert isinstance(x, dict)
        name = x["name"]
        return name in self.tools and x["arguments"] in self.tools[name]

    def seed(self, seed: int | None = None) -> list[int]:
        own_seed = cast(int, super().seed(seed))
        child_seeds: list[int] = []
        for space in self.tools.values():
            child_seed = int(self.np_random.integers(0, np.iinfo(np.int32).max))
            space.seed(child_seed)
            child_seeds.append(child_seed)
        super().seed(own_seed)
        return [own_seed, *child_seeds]

    def to_jsonable(self, sample_n: Sequence[ToolCall]) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for value in sample_n:
            validated = self.validate(value)
            name = validated["name"]
            if name not in self.tools:
                raise ValueError(f"unknown tool {name!r}")
            encoded_batch = self.tools[name].to_jsonable([validated["arguments"]])
            # Gym's Dict space represents a batch as a mapping of batches;
            # scalar/custom spaces generally represent it as a sequence.
            encoded_arguments = (
                encoded_batch
                if isinstance(self.tools[name], DictSpace)
                else encoded_batch[0]
            )
            item: dict[str, Any] = {
                "name": name,
                "arguments": encoded_arguments,
            }
            if "id" in validated:
                item["id"] = validated["id"]
            result.append(item)
        return result

    def from_jsonable(self, sample_n: Sequence[Any]) -> list[ToolCall]:
        result: list[ToolCall] = []
        for raw in sample_n:
            structural = self.validate(raw)
            name = structural["name"]
            if name not in self.tools:
                raise ValueError(f"unknown tool {name!r}")
            encoded_arguments = structural["arguments"]
            argument_space = self.tools[name]
            if isinstance(argument_space, DictSpace):
                arguments = argument_space.from_jsonable(
                    cast(dict[str, list[Any]], encoded_arguments)
                )[0]
            else:
                arguments = argument_space.from_jsonable([encoded_arguments])[0]
            value: ToolCall = {"name": name, "arguments": arguments}
            if "id" in structural:
                value["id"] = structural["id"]
            if value not in self:
                raise ValueError("decoded value is not contained in this space")
            result.append(value)
        return result

    def __repr__(self) -> str:
        return f"ToolCallSpace(tools={tuple(self.tools)!r})"


def call(
    schemas: Mapping[str, Space[Any]],
    *,
    include_id_in_samples: bool = False,
    seed: int | None = None,
) -> ToolCallSpace:
    return ToolCallSpace(
        schemas,
        include_id_in_samples=include_id_in_samples,
        seed=seed,
    )


def calls(
    schemas: Mapping[str, Space[Any]],
    *,
    include_id_in_samples: bool = False,
    stack: bool = False,
    seed: int | None = None,
) -> SequenceSpace:
    """Return a variable-length sequence of structured tool calls."""

    return SequenceSpace(
        call(schemas, include_id_in_samples=include_id_in_samples),
        stack=stack,
        seed=seed,
    )


__all__ = ["ToolCallSpace", "call", "calls"]
