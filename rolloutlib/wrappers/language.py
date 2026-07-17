"""Gymnasium wrappers for presenting environments to language agents."""

from __future__ import annotations

import copy
import json
from collections.abc import Callable, Sequence
from typing import Any, Generic, TypeVar, cast

import gymnasium as gym
from gymnasium.spaces import Dict as DictSpace
from gymnasium.spaces import Space

from .. import content
from ..spaces import messages, tools
from ..spaces.json import to_json_value
from ..types import Chat, ContentPart, Message, ToolCall


ObsT = TypeVar("ObsT")
ActT = TypeVar("ActT")


class ChatObservationWrapper(
    gym.ObservationWrapper[Chat, ActT, ObsT],
    gym.utils.RecordConstructorArgs,
    Generic[ObsT, ActT],
):
    """Transform native Gymnasium observations into language-agent chats.

    The wrapper uses Gymnasium's standard :class:`ObservationWrapper` contract.
    ``func`` may produce text-only or multimodal chats containing image and
    audio content parts.
    """

    def __init__(
        self,
        env: gym.Env[ObsT, ActT],
        func: Callable[[ObsT], Chat],
        *,
        observation_space: Space[Chat] | None = None,
    ) -> None:
        """Create a chat observation wrapper.

        Args:
            env: Native Gymnasium environment to wrap.
            func: Function mapping a native observation to a chat.
            observation_space: Space describing transformed chats. Defaults to
                a non-empty unconstrained-length ``ChatSpace``.

        Returns:
            ``None``.
        """
        gym.utils.RecordConstructorArgs.__init__(
            self,
            func=func,
            observation_space=observation_space,
        )
        gym.ObservationWrapper.__init__(self, env)
        self.func = func
        self.observation_space = observation_space or messages.chat(min_length=1)

    def observation(self, observation: ObsT) -> Chat:
        """Transform and validate one native observation.

        Args:
            observation: Observation returned by the wrapped environment.

        Returns:
            A validated chat suitable for a language-agent policy.

        Raises:
            ValueError: If ``func`` returns a value outside ``observation_space``.
        """
        chat = self.func(observation)
        if chat not in self.observation_space:
            raise ValueError(
                "transformed observation is outside the chat observation space"
            )
        return chat


class ToolCallActionWrapper(
    gym.ActionWrapper[ObsT, ToolCall, ActT],
    gym.utils.RecordConstructorArgs,
    Generic[ObsT, ActT],
):
    """Expose a native Gymnasium action space as a structured tool call.

    The policy-facing action has the form
    ``{"name": tool_name, "arguments": {argument_name: native_action}}``.
    Tool-call arguments are validated against the wrapped environment's native
    action space before :meth:`gymnasium.Env.step` is called.
    """

    def __init__(
        self,
        env: gym.Env[ObsT, ActT],
        *,
        tool_name: str = "step",
        argument_name: str = "action",
        tool_description: str | None = None,
        available_actions: Callable[[], Sequence[ActT]] | None = None,
    ) -> None:
        """Create a tool-call action wrapper.

        Args:
            env: Native Gymnasium environment to wrap.
            tool_name: Name exposed to the language agent.
            argument_name: Tool argument containing the native action.
            tool_description: Optional model-facing description of the action.
            available_actions: Optional callable returning actions currently
                accepted by the environment.

        Returns:
            ``None``.
        """
        if not tool_name:
            raise ValueError("tool_name must not be empty")
        if not argument_name:
            raise ValueError("argument_name must not be empty")
        gym.utils.RecordConstructorArgs.__init__(
            self,
            tool_name=tool_name,
            argument_name=argument_name,
            tool_description=tool_description,
            available_actions=available_actions,
        )
        gym.ActionWrapper.__init__(self, env)
        self.tool_name = tool_name
        self.argument_name = argument_name
        self.tool_description = tool_description
        self._available_actions = available_actions
        self.action_space = tools.call(
            {tool_name: DictSpace({argument_name: env.action_space})},
            descriptions=(
                {tool_name: tool_description}
                if tool_description is not None
                else None
            ),
        )

    def available_action_values(self) -> tuple[object, ...] | None:
        """Return currently available native actions as JSON values."""

        if self._available_actions is None:
            return None
        values = tuple(self._available_actions())
        return tuple(to_json_value(self.env.action_space, value) for value in values)

    def action(self, action: ToolCall) -> ActT:
        """Validate and unwrap one policy-facing tool call.

        Args:
            action: Tool call produced by the language-agent policy.

        Returns:
            The native action accepted by the wrapped environment.

        Raises:
            ValueError: If the tool call is outside ``action_space``.
        """
        try:
            decoded = cast(ToolCall, self.action_space.from_jsonable([action])[0])
        except (TypeError, ValueError) as error:
            raise ValueError(
                "action is outside the tool-call action space"
            ) from error
        arguments = cast(dict[str, object], decoded["arguments"])
        native_action = cast(ActT, arguments[self.argument_name])
        available = self.available_action_values()
        if available is not None:
            encoded = to_json_value(self.env.action_space, native_action)
            if encoded not in available:
                raise ValueError("action is not currently available")
        return native_action


class ChatHistoryWrapper(
    gym.Wrapper[Chat, ToolCall, Chat, ToolCall],
    gym.utils.RecordConstructorArgs,
):
    """Expose episode-scoped chat history as the current observation.

    The wrapper appends each tool-call action and subsequent chat observation.
    Earlier media is replaced with short text placeholders by default so image
    and audio payloads do not grow quadratically across a rollout.
    """

    def __init__(
        self,
        env: gym.Env[Chat, ToolCall],
        *,
        max_messages: int | None = None,
        retain_media: str = "latest",
    ) -> None:
        """Create a history wrapper around a chat/tool-call environment."""

        if max_messages is not None and max_messages < 1:
            raise ValueError("max_messages must be at least 1")
        if retain_media not in ("all", "latest"):
            raise ValueError("retain_media must be 'all' or 'latest'")
        gym.utils.RecordConstructorArgs.__init__(
            self,
            max_messages=max_messages,
            retain_media=retain_media,
        )
        gym.Wrapper.__init__(self, env)
        self.max_messages = max_messages
        self.retain_media = retain_media
        self.observation_space = messages.chat(
            min_length=1,
            max_length=max_messages,
        )
        self._history: Chat = []
        self._step_index = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Chat, dict[str, Any]]:
        """Reset the inner environment and clear episode history."""

        observation, info = self.env.reset(seed=seed, options=options)
        self._history = copy.deepcopy(observation)
        self._step_index = 0
        return self._snapshot(), info

    def step(
        self,
        action: ToolCall,
    ) -> tuple[Chat, float, bool, bool, dict[str, Any]]:
        """Append an action and the resulting observation to chat history."""

        observation, reward, terminated, truncated, info = self.env.step(action)
        recorded_action = cast(
            ToolCall,
            self.action_space.to_jsonable([copy.deepcopy(action)])[0],
        )
        recorded_action.setdefault("id", f"call_{self._step_index}")
        self._history.append(
            {
                "role": "assistant",
                "content": f"Action: {json.dumps(recorded_action, ensure_ascii=False)}",
                "tool_calls": [recorded_action],
            }
        )
        has_system = any(message["role"] == "system" for message in self._history)
        self._history.extend(
            copy.deepcopy(
                [
                    message
                    for message in observation
                    if message["role"] != "system" or not has_system
                ]
            )
        )
        self._step_index += 1
        return self._snapshot(), float(reward), terminated, truncated, info

    def _snapshot(self) -> Chat:
        history = copy.deepcopy(self._history)
        if self.retain_media == "latest" and len(history) > 1:
            history[:-1] = [self._without_media(message) for message in history[:-1]]
        if self.max_messages is not None and len(history) > self.max_messages:
            leading_system = (
                [history[0]]
                if self.max_messages > 1 and history[0]["role"] == "system"
                else []
            )
            tail_length = self.max_messages - len(leading_system)
            history = leading_system + (history[-tail_length:] if tail_length else [])
        if history not in self.observation_space:
            raise ValueError("chat history is outside the observation space")
        return history

    @staticmethod
    def _without_media(message: Message) -> Message:
        content_value = message["content"]
        if isinstance(content_value, str):
            return message
        resolved: list[ContentPart] = []
        for part in content_value:
            if part["type"] == "text":
                resolved.append(part)
            elif part["type"] == "image":
                label = part.get("alt", "image")
                resolved.append({"type": "text", "text": f"[Earlier image: {label}]"})
            else:
                resolved.append({"type": "text", "text": "[Earlier audio]"})
        message["content"] = resolved
        return message


def wrap_language_env(
    env: gym.Env[ObsT, ActT],
    func: Callable[[ObsT], Chat] | None = None,
    *,
    state: Callable[[ObsT], object] | None = None,
    image: Callable[[ObsT], Any] | None = None,
    image_alt: str = "Current environment observation",
    audio: Callable[[ObsT], Any] | None = None,
    audio_format: str = "wav",
    audio_sample_rate: int | None = None,
    include_render: bool = False,
    instructions: str | None = None,
    observation_space: Space[Chat] | None = None,
    tool_name: str = "step",
    argument_name: str = "action",
    tool_description: str | None = None,
    available_actions: Callable[[], Sequence[ActT]] | None = None,
    history: int | None = None,
    retain_media: str = "latest",
) -> gym.Env[Chat, ToolCall]:
    """Present a native Gymnasium environment to a language-agent policy.

    This convenience function composes :class:`ChatObservationWrapper` and
    :class:`ToolCallActionWrapper`. Use the wrapper classes directly when an
    application needs to insert other Gymnasium wrappers between them.

    Args:
        env: Native Gymnasium environment to wrap.
        func: Optional function mapping native observations to complete chats.
            When omitted, rolloutlib constructs a multimodal user message.
        state: Optional function selecting the text/JSON state to present.
        image: Optional function selecting an image URL, bytes, image, or array.
        image_alt: Alternative text for the selected image.
        audio: Optional function selecting an audio URL, bytes, or sample array.
        audio_format: Format used for selected audio.
        audio_sample_rate: Sample rate used for audio arrays.
        include_render: Include ``env.render()`` as an image.
        instructions: Optional system message included in each observation.
        observation_space: Space describing transformed chats.
        tool_name: Name of the policy-facing action tool.
        argument_name: Tool argument containing the native action.
        tool_description: Optional model-facing description of the action.
        available_actions: Optional callable returning currently legal actions.
        history: Optional maximum number of messages retained across the episode.
        retain_media: Whether history retains all media or only the latest media.

    Returns:
        A Gymnasium environment with chat observations and tool-call actions.
    """
    ergonomic_options_used = any(
        option is not None for option in (state, image, audio, instructions)
    ) or include_render
    if func is not None and ergonomic_options_used:
        raise ValueError(
            "func cannot be combined with state, image, audio, instructions, "
            "or include_render"
        )
    if image is not None and include_render:
        raise ValueError("image and include_render cannot be combined")

    if func is None:
        state_func = state or (
            lambda observation: to_json_value(env.observation_space, observation)
        )

        def make_chat(observation: ObsT) -> Chat:
            selected_state = state_func(observation)
            parts: list[ContentPart] = [
                (
                    content.text(selected_state)
                    if isinstance(selected_state, str)
                    else content.json(cast(Any, selected_state))
                )
            ]
            selected_image = (
                env.render() if include_render else image(observation) if image else None
            )
            if selected_image is not None:
                parts.append(content.image(selected_image, alt=image_alt))
            if audio is not None:
                selected_audio = audio(observation)
                if selected_audio is not None:
                    parts.append(
                        content.audio(
                            selected_audio,
                            format=audio_format,
                            sample_rate=audio_sample_rate,
                        )
                    )
            chat: Chat = []
            if instructions is not None:
                chat.append({"role": "system", "content": instructions})
            chat.append({"role": "user", "content": parts})
            return chat

        func = make_chat
        if observation_space is None:
            length = 2 if instructions is not None else 1
            observation_space = messages.chat(
                min_length=length,
                max_length=length,
            )

    observed = ChatObservationWrapper(
        env,
        func,
        observation_space=observation_space,
    )
    wrapped: gym.Env[Chat, ToolCall] = ToolCallActionWrapper(
        observed,
        tool_name=tool_name,
        argument_name=argument_name,
        tool_description=tool_description,
        available_actions=available_actions,
    )
    if history is not None:
        wrapped = ChatHistoryWrapper(
            wrapped,
            max_messages=history,
            retain_media=retain_media,
        )
    return wrapped


__all__ = [
    "ChatObservationWrapper",
    "ChatHistoryWrapper",
    "ToolCallActionWrapper",
    "wrap_language_env",
]
