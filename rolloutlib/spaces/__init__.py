"""Role-neutral Gymnasium spaces for language-agent values."""

from . import compatibility, json, messages, text, tokens, tools
from .compatibility import (
    check_space_compatibility,
    check_space_value,
    is_space_subset,
)
from .json import from_json_value, to_json_schema, to_json_value
from ._pydantic import PydanticSpace
from .messages import ChatSpace, MessageSpace
from .text import TextSpace
from .tools import ToolCallSpace

# Gymnasium-style short names for the common domain spaces. The longer names
# remain available when clarity is preferable.
Text = TextSpace
Message = MessageSpace
Chat = ChatSpace
ToolCall = ToolCallSpace

__all__ = [
    "ChatSpace",
    "Chat",
    "Message",
    "MessageSpace",
    "PydanticSpace",
    "Text",
    "TextSpace",
    "ToolCall",
    "ToolCallSpace",
    "check_space_compatibility",
    "check_space_value",
    "compatibility",
    "is_space_subset",
    "messages",
    "json",
    "from_json_value",
    "text",
    "to_json_schema",
    "to_json_value",
    "tokens",
    "tools",
]
