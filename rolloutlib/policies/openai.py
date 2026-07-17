"""OpenAI Responses API policies for chat and tool-call Gymnasium spaces."""

from __future__ import annotations

import importlib
import json
from collections.abc import Callable, Mapping
from typing import Any, cast

import gymnasium as gym

from ..rollouts import PolicyOutput
from ..spaces import ToolCallSpace, to_json_schema
from ..types import Chat, ToolCall


AvailableActions = Callable[[], tuple[object, ...] | None]
_SUPPORTED_SCHEMA_KEYS = {
    "$defs",
    "$ref",
    "additionalProperties",
    "anyOf",
    "const",
    "description",
    "enum",
    "exclusiveMaximum",
    "exclusiveMinimum",
    "format",
    "items",
    "maxItems",
    "maximum",
    "minItems",
    "minimum",
    "multipleOf",
    "pattern",
    "properties",
    "required",
    "type",
}


def _strict_schema(value: Any) -> Any:
    if isinstance(value, list):
        return [_strict_schema(item) for item in value]
    if not isinstance(value, dict):
        return value
    resolved: dict[str, Any] = {}
    for key, item in value.items():
        if key not in _SUPPORTED_SCHEMA_KEYS:
            continue
        if key in ("properties", "$defs") and isinstance(item, dict):
            resolved[key] = {
                child_key: _strict_schema(child)
                for child_key, child in item.items()
            }
        else:
            resolved[key] = _strict_schema(item)
    return resolved


def to_openai_input(
    chat: Chat,
    *,
    image_detail: str = "auto",
) -> list[dict[str, Any]]:
    """Translate backend-neutral rolloutlib messages into Responses input."""

    messages: list[dict[str, Any]] = []
    for message in chat:
        content = message["content"]
        if isinstance(content, str):
            resolved_content: str | list[dict[str, Any]] = content
        else:
            resolved_content = []
            for part in content:
                if part["type"] == "text":
                    resolved_content.append(
                        {"type": "input_text", "text": part["text"]}
                    )
                elif part["type"] == "image":
                    resolved_content.append(
                        {
                            "type": "input_image",
                            "image_url": part["url"],
                            "detail": image_detail,
                        }
                    )
                else:
                    raise ValueError(
                        "OpenAI Responses input does not accept rolloutlib audio "
                        "content; use a compatible Realtime policy"
                    )
        messages.append({"role": message["role"], "content": resolved_content})
    return messages


def _environment_available_actions(
    environment: gym.Env[Chat, ToolCall],
) -> AvailableActions | None:
    try:
        value = environment.get_wrapper_attr("available_action_values")
    except AttributeError:
        return None
    return cast(AvailableActions, value) if callable(value) else None


class _OpenAIResponsesPolicyBase:
    def __init__(
        self,
        client: Any,
        model: str,
        action_space: ToolCallSpace,
        *,
        instructions: str | None = None,
        reasoning: Mapping[str, Any] | None = None,
        max_output_tokens: int | None = None,
        image_detail: str = "auto",
        store: bool = False,
        request_options: Mapping[str, Any] | None = None,
        available_actions: AvailableActions | None = None,
        available_argument: str | None = None,
    ) -> None:
        if not model:
            raise ValueError("model must not be empty")
        if image_detail not in ("auto", "low", "high"):
            raise ValueError("image_detail must be auto, low, or high")
        self.client = client
        self.model = model
        self.action_space = action_space
        self.instructions = instructions
        self.reasoning = dict(reasoning) if reasoning is not None else None
        self.max_output_tokens = max_output_tokens
        self.image_detail = image_detail
        self.store = store
        self.request_options = dict(request_options or {})
        self.available_actions = available_actions
        self.available_argument = available_argument

    @classmethod
    def _from_env_options(
        cls,
        environment: gym.Env[Chat, ToolCall],
    ) -> tuple[ToolCallSpace, AvailableActions | None, str | None]:
        action_space = environment.action_space
        if not isinstance(action_space, ToolCallSpace):
            raise TypeError(
                "OpenAIResponsesPolicy requires a rolloutlib ToolCallSpace; "
                "wrap the environment with wrap_language_env"
            )
        available_actions = _environment_available_actions(environment)
        try:
            argument_name = environment.get_wrapper_attr("argument_name")
        except AttributeError:
            argument_name = None
        return (
            action_space,
            available_actions,
            argument_name if isinstance(argument_name, str) else None,
        )

    def _tools(self) -> list[dict[str, Any]]:
        definitions: list[dict[str, Any]] = []
        available = self.available_actions() if self.available_actions else None
        for name, argument_space in self.action_space.tool_schemas.items():
            parameters = cast(
                dict[str, Any],
                _strict_schema(to_json_schema(argument_space)),
            )
            if parameters.get("type") != "object":
                raise TypeError(
                    f"tool {name!r} must use a gymnasium.spaces.Dict argument space"
                )
            if (
                available is not None
                and self.available_argument is not None
                and len(self.action_space.tool_schemas) == 1
            ):
                properties = cast(dict[str, dict[str, Any]], parameters["properties"])
                if self.available_argument in properties:
                    properties[self.available_argument] = {
                        **properties[self.available_argument],
                        "enum": list(available),
                    }
            definition: dict[str, Any] = {
                "type": "function",
                "name": name,
                "parameters": parameters,
                "strict": True,
            }
            description = self.action_space.descriptions.get(name)
            if description is not None:
                definition["description"] = description
            definitions.append(definition)
        return definitions

    def _request(self, observation: Chat) -> dict[str, Any]:
        tools = self._tools()
        request: dict[str, Any] = {
            "model": self.model,
            "input": to_openai_input(
                observation,
                image_detail=self.image_detail,
            ),
            "tools": tools,
            "tool_choice": (
                {"type": "function", "name": tools[0]["name"]}
                if len(tools) == 1
                else "required"
            ),
            "parallel_tool_calls": False,
            "store": self.store,
            **self.request_options,
        }
        if self.instructions is not None:
            request["instructions"] = self.instructions
        if self.reasoning is not None:
            request["reasoning"] = self.reasoning
        if self.max_output_tokens is not None:
            request["max_output_tokens"] = self.max_output_tokens
        return request

    def _output(self, response: Any) -> PolicyOutput[ToolCall]:
        reasoning_summaries: list[str] = []
        for item in response.output:
            if getattr(item, "type", None) != "reasoning":
                continue
            for summary in getattr(item, "summary", ()) or ():
                text = getattr(summary, "text", None)
                if isinstance(text, str) and text:
                    reasoning_summaries.append(text)
        calls = [
            item
            for item in response.output
            if getattr(item, "type", None) == "function_call"
        ]
        if len(calls) != 1:
            raise ValueError(f"expected exactly one function call, got {response.output}")
        call = calls[0]
        raw_action = {
            "name": call.name,
            "arguments": json.loads(call.arguments),
            "id": call.call_id,
        }
        action = self.action_space.from_jsonable([raw_action])[0]
        info: dict[str, Any] = {
            "response_id": response.id,
            "model": response.model,
        }
        reasoning = getattr(response, "reasoning", None)
        if reasoning is not None:
            info["reasoning_effort"] = getattr(reasoning, "effort", None)
        if reasoning_summaries:
            info["reasoning_summary"] = "\n\n".join(reasoning_summaries)
        usage = getattr(response, "usage", None)
        if usage is not None:
            info["usage"] = (
                usage.model_dump(mode="json")
                if hasattr(usage, "model_dump")
                else usage
            )
        return PolicyOutput(action=action, info=info)


class OpenAIResponsesPolicy(_OpenAIResponsesPolicyBase):
    """Synchronous OpenAI Responses policy producing one Gymnasium action."""

    def __init__(
        self,
        model: str,
        action_space: ToolCallSpace,
        *,
        client: Any | None = None,
        **options: Any,
    ) -> None:
        if client is None:
            client = importlib.import_module("openai").OpenAI()
        super().__init__(client, model, action_space, **options)

    @classmethod
    def from_env(
        cls,
        environment: gym.Env[Chat, ToolCall],
        *,
        model: str,
        client: Any | None = None,
        **options: Any,
    ) -> OpenAIResponsesPolicy:
        """Create a policy bound to a wrapped environment's action space."""

        action_space, available_actions, available_argument = cls._from_env_options(
            environment
        )
        return cls(
            model,
            action_space,
            client=client,
            available_actions=available_actions,
            available_argument=available_argument,
            **options,
        )

    def __call__(self, observation: Chat) -> PolicyOutput[ToolCall]:
        response = self.client.responses.create(**self._request(observation))
        return self._output(response)


class AsyncOpenAIResponsesPolicy(_OpenAIResponsesPolicyBase):
    """Asynchronous OpenAI Responses policy producing one Gymnasium action."""

    def __init__(
        self,
        model: str,
        action_space: ToolCallSpace,
        *,
        client: Any | None = None,
        **options: Any,
    ) -> None:
        if client is None:
            client = importlib.import_module("openai").AsyncOpenAI()
        super().__init__(client, model, action_space, **options)

    @classmethod
    def from_env(
        cls,
        environment: gym.Env[Chat, ToolCall],
        *,
        model: str,
        client: Any | None = None,
        **options: Any,
    ) -> AsyncOpenAIResponsesPolicy:
        """Create an async policy bound to a wrapped environment's action space."""

        action_space, available_actions, available_argument = cls._from_env_options(
            environment
        )
        return cls(
            model,
            action_space,
            client=client,
            available_actions=available_actions,
            available_argument=available_argument,
            **options,
        )

    async def __call__(self, observation: Chat) -> PolicyOutput[ToolCall]:
        response = await self.client.responses.create(**self._request(observation))
        return self._output(response)


__all__ = [
    "AsyncOpenAIResponsesPolicy",
    "OpenAIResponsesPolicy",
    "to_openai_input",
]
