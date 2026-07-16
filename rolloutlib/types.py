"""Runtime-light structured values used by rolloutlib environments.

The types in this module deliberately describe ordinary dictionaries and lists.
Pydantic is used to validate those values at space boundaries; callers do not
need to construct framework-specific model objects.
"""

from __future__ import annotations

from typing import Literal, NotRequired, TypeAlias, TypedDict

from pydantic import ConfigDict, with_config


# The common roles are documented defaults, not a closed protocol. Providers
# and applications may use additional role names and constrain them through a
# MessageSpace configuration.
Role: TypeAlias = str


@with_config(ConfigDict(extra="forbid"))
class ToolCall(TypedDict):
    """A structured invocation of a named tool."""

    name: str
    arguments: dict[str, object]
    id: NotRequired[str]


@with_config(ConfigDict(extra="forbid"))
class TextContentPart(TypedDict):
    type: Literal["text"]
    text: str


@with_config(ConfigDict(extra="forbid"))
class ImageContentPart(TypedDict):
    """A backend-neutral reference to image content."""

    type: Literal["image"]
    url: str
    alt: NotRequired[str]


@with_config(ConfigDict(extra="forbid"))
class AudioContentPart(TypedDict):
    """A backend-neutral reference to audio content."""

    type: Literal["audio"]
    url: str
    format: NotRequired[str]


ContentPart: TypeAlias = TextContentPart | ImageContentPart | AudioContentPart


@with_config(ConfigDict(extra="forbid"))
class Message(TypedDict):
    """A role-tagged chat message with optional tool-call metadata."""

    role: Role
    content: str | list[ContentPart]
    tool_calls: NotRequired[list[ToolCall]]
    tool_call_id: NotRequired[str]
    name: NotRequired[str]


Chat: TypeAlias = list[Message]


__all__ = [
    "AudioContentPart",
    "Chat",
    "ContentPart",
    "ImageContentPart",
    "Message",
    "Role",
    "TextContentPart",
    "ToolCall",
]
