"""Spaces for structured tool calls."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

import numpy as np
from gymnasium import Space
from gymnasium.spaces import Sequence as SequenceSpace

from rolloutlib.types import ToolCall

from ._pydantic import PydanticSpace
from .json import from_json_value, to_json_value


class ToolCallSpace(PydanticSpace[ToolCall]):
    """Validate a tool name and its arguments with a tool-specific Gym space."""

    def __init__(
        self,
        tools: Mapping[str, Space[Any]],
        *,
        descriptions: Mapping[str, str] | None = None,
        include_id_in_samples: bool = False,
        seed: int | None = None,
    ) -> None:
        """Initialize a space for named structured tool calls.

        Args:
            tools: Mapping from tool names to argument spaces.
            descriptions: Optional model-facing description for each tool.
            include_id_in_samples: Whether sampled calls include generated IDs.
            seed: Optional random seed.

        Returns:
            ``None``.
        """
        if not tools:
            raise ValueError("at least one tool schema is required")
        if any(not isinstance(name, str) or not name for name in tools):
            raise ValueError("tool names must be non-empty strings")
        if any(not isinstance(space, Space) for space in tools.values()):
            raise TypeError("each tool schema must be a gymnasium Space")
        if descriptions is not None and not set(descriptions).issubset(tools):
            raise ValueError("descriptions contain unknown tool names")
        self.tools = dict(tools)
        self.descriptions = dict(descriptions or {})
        self.include_id_in_samples = include_id_in_samples
        super().__init__(ToolCall, sampler=self._sample_call, seed=seed)

    @property
    def tool_schemas(self) -> Mapping[str, Space[Any]]:
        """Return the configured tool-name to argument-space mapping.

        Returns:
            Read-only view of the tool schemas.
        """
        return self.tools

    def _sample_call(self, rng: np.random.Generator) -> ToolCall:
        """Sample a tool name and arguments from its schema.

        Args:
            rng: NumPy random generator supplied by the space.

        Returns:
            Sampled structured tool call.
        """
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
        """Check tool-name and argument-space membership.

        Args:
            x: Candidate tool call.

        Returns:
            ``True`` when the call names a known tool with valid arguments.
        """
        if not super().contains(x):
            return False
        assert isinstance(x, dict)
        name = x["name"]
        return name in self.tools and x["arguments"] in self.tools[name]

    def seed(self, seed: int | None = None) -> list[int]:
        """Seed this space and every tool argument space.

        Args:
            seed: Optional seed for this space.

        Returns:
            The parent seed followed by one seed per tool schema.
        """
        own_seed = cast(int, super().seed(seed))
        child_seeds: list[int] = []
        for space in self.tools.values():
            child_seed = int(self.np_random.integers(0, np.iinfo(np.int32).max))
            space.seed(child_seed)
            child_seeds.append(child_seed)
        super().seed(own_seed)
        return [own_seed, *child_seeds]

    def to_jsonable(self, sample_n: Sequence[ToolCall]) -> list[dict[str, Any]]:
        """Serialize tool calls using each tool's argument space.

        Args:
            sample_n: Tool calls to serialize.

        Returns:
            JSON-compatible tool call mappings.
        """
        result: list[dict[str, Any]] = []
        for value in sample_n:
            validated = self.validate(value)
            name = validated["name"]
            if name not in self.tools:
                raise ValueError(f"unknown tool {name!r}")
            item: dict[str, Any] = {
                "name": name,
                "arguments": to_json_value(
                    self.tools[name], validated["arguments"]
                ),
            }
            if "id" in validated:
                item["id"] = validated["id"]
            result.append(item)
        return result

    def from_jsonable(self, sample_n: Sequence[Any]) -> list[ToolCall]:
        """Deserialize and validate serialized tool calls.

        Args:
            sample_n: JSON-compatible tool call mappings.

        Returns:
            Validated tool calls with decoded arguments.
        """
        result: list[ToolCall] = []
        for raw in sample_n:
            structural = self.validate(raw)
            name = structural["name"]
            if name not in self.tools:
                raise ValueError(f"unknown tool {name!r}")
            argument_space = self.tools[name]
            arguments = from_json_value(
                argument_space,
                cast(Any, structural["arguments"]),
            )
            value: ToolCall = {"name": name, "arguments": arguments}
            if "id" in structural:
                value["id"] = structural["id"]
            if value not in self:
                raise ValueError("decoded value is not contained in this space")
            result.append(value)
        return result

    def __repr__(self) -> str:
        """Return a concise representation of the tool-call space.

        Returns:
            Human-readable representation listing configured tool names.
        """
        return f"ToolCallSpace(tools={tuple(self.tools)!r})"


def call(
    schemas: Mapping[str, Space[Any]],
    *,
    descriptions: Mapping[str, str] | None = None,
    include_id_in_samples: bool = False,
    seed: int | None = None,
) -> ToolCallSpace:
    """Construct a space for one structured tool call.

    Args:
        schemas: Mapping from tool names to argument spaces.
        descriptions: Optional model-facing description for each tool.
        include_id_in_samples: Whether sampled calls include generated IDs.
        seed: Optional random seed.

    Returns:
        Configured ``ToolCallSpace`` instance.
    """
    return ToolCallSpace(
        schemas,
        descriptions=descriptions,
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
    """Construct a variable-length sequence of structured tool calls.

    Args:
        schemas: Mapping from tool names to argument spaces.
        include_id_in_samples: Whether sampled calls include generated IDs.
        stack: Whether sampled sequences should be stacked when possible.
        seed: Optional random seed.

    Returns:
        Gymnasium ``Sequence`` space containing tool calls.
    """

    return SequenceSpace(
        call(schemas, include_id_in_samples=include_id_in_samples),
        stack=stack,
        seed=seed,
    )


__all__ = ["ToolCallSpace", "call", "calls"]
