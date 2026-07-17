from __future__ import annotations

from typing import Any, cast

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from rolloutlib import rollout, spaces
from rolloutlib.wrappers import (
    ChatHistoryWrapper,
    ChatObservationWrapper,
    ToolCallActionWrapper,
    wrap_language_env,
)
from rolloutlib.types import Chat, ToolCall


FloatArray = np.ndarray[Any, np.dtype[np.float32]]


class SensoryEnv(gym.Env[dict[str, Any], int]):
    """Small native Gymnasium environment with multimodal observation references."""

    action_space = gym.spaces.Discrete(3)
    observation_space = gym.spaces.Dict(
        {
            "audio_url": spaces.text.text(max_length=200),
            "image_url": spaces.text.text(max_length=200),
            "text": spaces.text.text(max_length=200),
        }
    )

    def __init__(self) -> None:
        super().__init__()
        self.last_action: int | None = None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        super().reset(seed=seed)
        return self._observation("ready"), {"options": options}

    def step(
        self, action: int
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        assert action in self.action_space
        self.last_action = int(action)
        return self._observation("done"), float(action), True, False, {}

    @staticmethod
    def _observation(text: str) -> dict[str, Any]:
        return {
            "audio_url": "https://example.test/observation.wav",
            "image_url": "https://example.test/observation.png",
            "text": text,
        }


class ContinuousActionEnv(gym.Env[FloatArray, FloatArray]):
    action_space = gym.spaces.Box(
        low=-1.0,
        high=1.0,
        shape=(3,),
        dtype=np.float32,
    )
    observation_space = gym.spaces.Box(
        low=-1.0,
        high=1.0,
        shape=(1,),
        dtype=np.float32,
    )

    def __init__(self) -> None:
        super().__init__()
        self.last_action: FloatArray | None = None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[FloatArray, dict[str, Any]]:
        super().reset(seed=seed)
        return np.zeros(1, dtype=np.float32), {}

    def step(
        self,
        action: FloatArray,
    ) -> tuple[
        FloatArray,
        float,
        bool,
        bool,
        dict[str, Any],
    ]:
        assert isinstance(action, np.ndarray)
        assert action in self.action_space
        self.last_action = action
        return np.ones(1, dtype=np.float32), 0.0, True, False, {}


def make_language_env(
    inner: SensoryEnv,
) -> gym.Env[Chat, ToolCall]:
    """Compose standard Gymnasium wrappers into a language-agent environment."""

    chat_space = spaces.messages.chat(min_length=1, max_length=1)

    def to_chat(observation: dict[str, Any]) -> Chat:
        return [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": cast(str, observation["text"])},
                    {
                        "type": "image",
                        "url": cast(str, observation["image_url"]),
                        "alt": "Current environment observation",
                    },
                    {
                        "type": "audio",
                        "url": cast(str, observation["audio_url"]),
                        "format": "wav",
                    },
                ],
            }
        ]

    return wrap_language_env(
        inner,
        to_chat,
        observation_space=chat_space,
    )


def test_gymnasium_language_wrappers_expose_language_agent_spaces() -> None:
    inner = SensoryEnv()
    env = make_language_env(inner)

    assert isinstance(env, ToolCallActionWrapper)
    assert isinstance(env, gym.ActionWrapper)
    assert isinstance(env.env, ChatObservationWrapper)
    assert isinstance(env.env, gym.ObservationWrapper)

    observation, info = env.reset(seed=7, options={"source": "test"})
    action: ToolCall = {"name": "step", "arguments": {"action": 2}}
    next_observation, reward, terminated, truncated, step_info = env.step(action)

    assert observation in env.observation_space
    assert action in env.action_space
    assert next_observation in env.observation_space
    assert observation[0]["content"] == [
        {"type": "text", "text": "ready"},
        {
            "type": "image",
            "url": "https://example.test/observation.png",
            "alt": "Current environment observation",
        },
        {
            "type": "audio",
            "url": "https://example.test/observation.wav",
            "format": "wav",
        },
    ]
    assert info == {"options": {"source": "test"}}
    assert (reward, terminated, truncated, step_info) == (2.0, True, False, {})
    assert inner.last_action == 2
    assert env.unwrapped is inner


def test_gymnasium_language_wrappers_work_with_rollouts() -> None:
    inner = SensoryEnv()
    action: ToolCall = {"name": "step", "arguments": {"action": 1}}

    trajectory = rollout(make_language_env(inner), lambda observation: action)

    assert trajectory.initial_observation[0]["content"][0] == {
        "type": "text",
        "text": "ready",
    }
    assert trajectory.steps[0].action == action
    assert trajectory.steps[0].next_observation[0]["content"][0] == {
        "type": "text",
        "text": "done",
    }
    assert trajectory.total_reward == 1.0
    assert inner.last_action == 1


@pytest.mark.parametrize(
    "action",
    [
        {"name": "missing", "arguments": {"action": 1}},
        {"name": "step", "arguments": {"action": 3}},
    ],
)
def test_language_action_wrapper_rejects_invalid_tool_calls(
    action: ToolCall,
) -> None:
    env = make_language_env(SensoryEnv())
    env.reset()

    with pytest.raises(ValueError, match="outside the tool-call action space"):
        env.step(action)


def test_language_wrappers_support_custom_tool_and_argument_names() -> None:
    inner = SensoryEnv()
    observed = ChatObservationWrapper(
        inner,
        lambda observation: [
            {"role": "user", "content": cast(str, observation["text"])}
        ],
    )
    env = ToolCallActionWrapper(
        observed,
        tool_name="choose",
        argument_name="value",
    )
    action: ToolCall = {"name": "choose", "arguments": {"value": 2}}

    observation, _ = env.reset()
    env.step(action)

    assert observation in env.observation_space
    assert action in env.action_space
    assert inner.last_action == 2


def test_language_action_wrapper_decodes_json_arrays_to_native_box_actions() -> None:
    inner = ContinuousActionEnv()
    env = wrap_language_env(
        inner,
        tool_name="rebalance",
        argument_name="signals",
    )

    env.reset()
    env.step(
        {
            "name": "rebalance",
            "arguments": {"signals": [0.5, 0.0, -0.5]},
        }
    )

    assert inner.last_action is not None
    assert inner.last_action.dtype == np.float32
    np.testing.assert_array_equal(
        inner.last_action,
        np.array([0.5, 0.0, -0.5], dtype=np.float32),
    )


def test_chat_history_serializes_native_box_actions_as_json_arrays() -> None:
    env = wrap_language_env(
        ContinuousActionEnv(),
        tool_name="rebalance",
        argument_name="signals",
        history=4,
    )

    env.reset()
    observation, *_ = env.step(
        {
            "name": "rebalance",
            "arguments": {
                "signals": np.array([0.5, 0.0, -0.5], dtype=np.float32)
            },
        }
    )

    assert observation[1].get("tool_calls") == [
        {
            "name": "rebalance",
            "arguments": {"signals": [0.5, 0.0, -0.5]},
            "id": "call_0",
        }
    ]


def test_chat_observation_wrapper_validates_transformed_observations() -> None:
    env = ChatObservationWrapper(SensoryEnv(), lambda observation: [])

    with pytest.raises(ValueError, match="outside the chat observation space"):
        env.reset()


def test_language_wrappers_retain_gymnasium_compatibility() -> None:
    with pytest.warns(UserWarning, match="different from the unwrapped version"):
        check_env(make_language_env(SensoryEnv()), skip_render_check=True)


def test_wrap_language_env_builds_json_and_multimodal_chat_without_chat_boilerplate() -> None:
    env = wrap_language_env(
        SensoryEnv(),
        state=lambda observation: {"status": observation["text"]},
        image=lambda observation: observation["image_url"],
        image_alt="Sensor view",
        audio=lambda observation: observation["audio_url"],
        instructions="Choose one action.",
        tool_description="Select a sensor action.",
    )

    observation, _ = env.reset()

    assert observation == [
        {"role": "system", "content": "Choose one action."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": '{"status":"ready"}'},
                {
                    "type": "image",
                    "url": "https://example.test/observation.png",
                    "alt": "Sensor view",
                },
                {
                    "type": "audio",
                    "url": "https://example.test/observation.wav",
                    "format": "wav",
                },
            ],
        },
    ]
    assert isinstance(env.action_space, spaces.ToolCallSpace)
    assert env.action_space.descriptions == {
        "step": "Select a sensor action."
    }


def test_wrap_language_env_defaults_to_json_serializing_native_observations() -> None:
    env = wrap_language_env(SensoryEnv())

    observation, _ = env.reset()

    assert observation[0]["role"] == "user"
    content = observation[0]["content"]
    assert isinstance(content, list)
    assert content[0]["type"] == "text"
    assert '"text":"ready"' in content[0]["text"]


def test_existing_gymnasium_environment_works_without_custom_conversion_code() -> None:
    env = wrap_language_env(gym.make("CartPole-v1"))

    observation, _ = env.reset(seed=7)
    next_observation, reward, terminated, truncated, _ = env.step(
        {"name": "step", "arguments": {"action": 0}}
    )

    assert observation in env.observation_space
    assert next_observation in env.observation_space
    assert isinstance(reward, float)
    assert not terminated
    assert not truncated
    env.close()


def test_tool_action_wrapper_validates_currently_available_actions() -> None:
    current = [1, 2]
    env = ToolCallActionWrapper(
        ChatObservationWrapper(
            SensoryEnv(),
            lambda observation: [
                {"role": "user", "content": cast(str, observation["text"])}
            ],
        ),
        available_actions=lambda: current,
    )
    env.reset()

    env.step({"name": "step", "arguments": {"action": 2}})

    env.reset()
    with pytest.raises(ValueError, match="not currently available"):
        env.step({"name": "step", "arguments": {"action": 0}})


def test_chat_history_wrapper_records_actions_and_retains_only_latest_media() -> None:
    env = wrap_language_env(
        SensoryEnv(),
        state=lambda observation: observation["text"],
        image=lambda observation: observation["image_url"],
        history=4,
    )

    assert isinstance(env, ChatHistoryWrapper)
    initial, _ = env.reset()
    action: ToolCall = {"name": "step", "arguments": {"action": 1}}
    final, *_ = env.step(action)

    initial_content = initial[-1]["content"]
    assert isinstance(initial_content, list)
    assert any(part["type"] == "image" for part in initial_content)
    assert len(final) == 3
    earlier_content = final[0]["content"]
    assert isinstance(earlier_content, list)
    assert all(part["type"] != "image" for part in earlier_content)
    assert final[1]["role"] == "assistant"
    tool_calls = final[1].get("tool_calls")
    assert tool_calls is not None
    assert tool_calls[0]["arguments"] == {"action": 1}
    latest_content = final[-1]["content"]
    assert isinstance(latest_content, list)
    assert any(part["type"] == "image" for part in latest_content)
