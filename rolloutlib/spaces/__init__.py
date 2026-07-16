"""Role-neutral Gymnasium spaces for language-agent values."""

from . import messages, text, tokens, tools
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
    "messages",
    "text",
    "tokens",
    "tools",
]
