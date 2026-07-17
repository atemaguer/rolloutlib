from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env
from pydantic import ValidationError

from rolloutlib import spaces
from rolloutlib.types import Chat, ToolCall


def assert_json_round_trip(space: gym.Space[Any]) -> None:
    samples = [space.sample() for _ in range(4)]
    payload = space.to_jsonable(samples)
    restored = space.from_jsonable(payload)

    assert len(restored) == len(samples)
    assert all(value in space for value in restored)


def test_structured_spaces_use_ordinary_python_values() -> None:
    message_space = spaces.messages.message(seed=7)
    chat_space = spaces.messages.chat(message_space, min_length=1, seed=8)
    tool_space = spaces.tools.call(
        {"search": gym.spaces.Dict({"limit": gym.spaces.Discrete(5)})},
        seed=9,
    )
    message = message_space.sample()
    chat = chat_space.sample()
    tool_call = tool_space.sample()

    assert type(message) is dict
    assert type(chat) is list
    assert all(type(item) is dict for item in chat)
    assert type(tool_call) is dict
    assert type(tool_call["arguments"]) is dict


def test_message_validation_is_strict_and_forbids_extra_fields() -> None:
    space = spaces.messages.message(roles=("user",))
    valid = {"role": "user", "content": "hello"}

    assert valid in space
    assert {"role": "assistant", "content": "hello"} not in space
    assert {"role": "user", "content": 123} not in space
    assert {"role": "user", "content": "hello", "unknown": True} not in space

    with pytest.raises(ValidationError):
        space.validate({"role": "user", "content": 123})


def test_message_roles_can_be_application_defined() -> None:
    space = spaces.messages.message(roles=("developer",))
    assert {"role": "developer", "content": "hello"} in space


def test_text_space_accepts_unicode_independently_of_sampling_alphabet() -> None:
    space = spaces.text.text(max_length=10, sample_alphabet="abc")
    assert "こんにちは" in space


def test_message_space_accepts_typed_content_parts_without_models() -> None:
    value = {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "look"},
            {"type": "image", "url": "https://example.test/image.png"},
        ],
        "tool_calls": [{"name": "search", "arguments": {"query": "rolloutlib"}}],
    }

    assert value in spaces.messages.message()


def test_tool_call_validation_is_strict_and_schema_aware() -> None:
    space = spaces.tools.call(
        {
            "search": gym.spaces.Dict(
                {
                    "limit": gym.spaces.Discrete(5),
                    "query": spaces.text.text(min_length=1, max_length=20),
                }
            )
        }
    )

    assert {
        "name": "search",
        "arguments": {"limit": 2, "query": "gymnasium"},
    } in space
    assert {
        "name": "missing",
        "arguments": {"limit": 2, "query": "gymnasium"},
    } not in space
    assert {
        "name": "search",
        "arguments": {"limit": "2", "query": "gymnasium"},
    } not in space
    assert {
        "name": "search",
        "arguments": {"limit": 2, "query": "gymnasium"},
        "extra": True,
    } not in space


def test_common_spaces_sample_values_they_contain_and_round_trip_json() -> None:
    token_id = spaces.tokens.id(32, seed=1)
    token_sequence = spaces.tokens.sequence(32, seed=2)
    message = spaces.messages.message(seed=3)
    chat = spaces.messages.chat(message, min_length=1, max_length=3, seed=4)
    tool_call = spaces.tools.call(
        {
            "search": gym.spaces.Dict(
                {
                    "limit": gym.spaces.Discrete(5),
                    "query": spaces.text.text(max_length=10),
                }
            )
        },
        include_id_in_samples=True,
        seed=5,
    )
    tool_sequence = spaces.tools.calls(
        {"search": gym.spaces.Dict({"query": spaces.text.text(max_length=10)})}
    )
    for space in (
        token_id,
        token_sequence,
        message,
        chat,
        tool_call,
        tool_sequence,
    ):
        assert space.sample() in space
        assert_json_round_trip(space)

    assert isinstance(spaces.Text(max_length=2), spaces.TextSpace)
    assert isinstance(spaces.Message(), spaces.MessageSpace)
    assert isinstance(spaces.Chat(), spaces.ChatSpace)
    assert isinstance(spaces.ToolCall({"search": gym.spaces.Dict({})}), spaces.ToolCallSpace)


def test_single_value_json_codecs_round_trip_nested_gymnasium_spaces() -> None:
    space = gym.spaces.Dict(
        {
            "choice": gym.spaces.Discrete(3),
            "coordinates": gym.spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(2,),
                dtype=np.float32,
            ),
            "label": gym.spaces.Text(min_length=1, max_length=8),
        }
    )
    value = {
        "choice": np.int64(2),
        "coordinates": np.array([0.25, -0.5], dtype=np.float32),
        "label": "move",
    }

    encoded = spaces.to_json_value(space, value)
    decoded = spaces.from_json_value(space, encoded)

    assert encoded == {
        "choice": 2,
        "coordinates": [0.25, -0.5],
        "label": "move",
    }
    assert decoded in space
    assert decoded["choice"] == 2
    np.testing.assert_array_equal(decoded["coordinates"], value["coordinates"])


def test_json_schema_describes_common_action_spaces() -> None:
    schema = spaces.to_json_schema(
        gym.spaces.Dict(
            {
                "choice": gym.spaces.Discrete(3),
                "label": spaces.text.text(min_length=1, max_length=8),
            }
        )
    )

    assert schema == {
        "type": "object",
        "properties": {
            "choice": {
                "type": "integer",
                "minimum": 0,
                "maximum": 2,
            },
            "label": {
                "type": "string",
                "minLength": 1,
                "maxLength": 8,
            },
        },
        "required": ["choice", "label"],
        "additionalProperties": False,
    }


def test_tool_call_space_json_uses_unbatched_argument_objects() -> None:
    space = spaces.tools.call(
        {
            "play": gym.spaces.Dict(
                {"move": gym.spaces.Text(min_length=4, max_length=5)}
            )
        }
    )
    value: ToolCall = {"name": "play", "arguments": {"move": "e2e4"}}

    encoded = space.to_jsonable([value])
    decoded = space.from_jsonable(encoded)

    assert encoded == [{"name": "play", "arguments": {"move": "e2e4"}}]
    assert decoded == [value]


def test_seeded_domain_spaces_are_reproducible() -> None:
    first = spaces.messages.chat(min_length=2, max_length=2, seed=123)
    second = spaces.messages.chat(min_length=2, max_length=2, seed=123)

    assert first.sample() == second.sample()


def test_structured_spaces_work_with_gymnasium_env_checker() -> None:
    class ToolEnv(gym.Env[Chat, ToolCall]):
        action_space = spaces.tools.call(
            {
                "search": gym.spaces.Dict(
                    {"query": spaces.text.text(min_length=1, max_length=32)}
                )
            },
            seed=1,
        )
        observation_space = spaces.messages.chat(
            min_length=0,
            max_length=2,
            seed=2,
        )

        def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
        ) -> tuple[Chat, dict[str, Any]]:
            super().reset(seed=seed)
            return [{"role": "user", "content": "Search"}], {}

        def step(
            self, action: ToolCall
        ) -> tuple[Chat, float, bool, bool, dict[str, Any]]:
            assert action in self.action_space
            return [], 1.0, True, False, {}

    check_env(ToolEnv(), skip_render_check=True)
