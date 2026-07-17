from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import gymnasium as gym

from rolloutlib import wrappers
from rolloutlib.policies.openai import OpenAIResponsesPolicy, to_openai_input
from rolloutlib.types import Chat


class SensoryEnv(gym.Env[dict[str, str], int]):
    action_space = gym.spaces.Discrete(3)
    observation_space = gym.spaces.Dict(
        {
            "image_url": gym.spaces.Text(
                max_length=200,
                charset="abcdefghijklmnopqrstuvwxyz:/.test",
            ),
            "text": gym.spaces.Text(
                max_length=200,
                charset="abcdefghijklmnopqrstuvwxyz",
            ),
        }
    )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, str], dict[str, Any]]:
        super().reset(seed=seed)
        return {
            "image_url": "https://example.test/observation.png",
            "text": "ready",
        }, {}

    def step(
        self, action: int
    ) -> tuple[dict[str, str], float, bool, bool, dict[str, Any]]:
        return {
            "image_url": "https://example.test/observation.png",
            "text": "done",
        }, float(action), True, False, {}


class FakeResponses:
    def __init__(self) -> None:
        self.request: dict[str, Any] | None = None

    def create(self, **request: Any) -> Any:
        self.request = request
        output = []
        if request.get("reasoning", {}).get("summary") is not None:
            output.append(
                SimpleNamespace(
                    type="reasoning",
                    summary=[
                        SimpleNamespace(
                            type="summary_text",
                            text="Compared the available actions and selected two.",
                        )
                    ],
                )
            )
        output.append(
            SimpleNamespace(
                type="function_call",
                name="choose",
                arguments=json.dumps({"value": 2}),
                call_id="call_test",
            )
        )
        return SimpleNamespace(
            id="resp_test",
            model=request["model"],
            reasoning=SimpleNamespace(
                effort=request.get("reasoning", {}).get("effort")
            ),
            usage=SimpleNamespace(
                model_dump=lambda **_: {"input_tokens": 10, "output_tokens": 2}
            ),
            output=output,
        )


def test_openai_responses_policy_derives_tools_and_parses_actions_from_environment() -> None:
    responses = FakeResponses()
    client = SimpleNamespace(responses=responses)
    available = [1, 2]
    env = wrappers.wrap_language_env(
        SensoryEnv(),
        state=lambda observation: observation["text"],
        image=lambda observation: observation["image_url"],
        tool_name="choose",
        argument_name="value",
        tool_description="Choose the next sensor action.",
        available_actions=lambda: available,
    )
    policy = OpenAIResponsesPolicy.from_env(
        env,
        client=client,
        model="test-model",
        instructions="Choose one action.",
        reasoning={"effort": "none"},
        image_detail="low",
        max_output_tokens=64,
    )
    observation, _ = env.reset()

    output = policy(observation)

    assert output.action == {
        "name": "choose",
        "arguments": {"value": 2},
        "id": "call_test",
    }
    assert output.info["response_id"] == "resp_test"
    assert responses.request is not None
    assert responses.request["tool_choice"] == {
        "type": "function",
        "name": "choose",
    }
    assert responses.request["parallel_tool_calls"] is False
    assert responses.request["store"] is False
    tool = responses.request["tools"][0]
    assert tool["description"] == "Choose the next sensor action."
    assert tool["parameters"]["properties"]["value"]["enum"] == [1, 2]
    assert responses.request["input"][0]["content"][1] == {
        "type": "input_image",
        "image_url": "https://example.test/observation.png",
        "detail": "low",
    }


def test_to_openai_input_rejects_audio_with_a_specific_error() -> None:
    chat: Chat = [
        {
            "role": "user",
            "content": [
                {
                    "type": "audio",
                    "url": "data:audio/wav;base64,AAAA",
                    "format": "wav",
                }
            ],
        }
    ]

    try:
        to_openai_input(chat)
    except ValueError as error:
        assert "Realtime policy" in str(error)
    else:
        raise AssertionError("expected audio input to be rejected")


def test_openai_responses_policy_records_reasoning_summaries() -> None:
    responses = FakeResponses()
    client = SimpleNamespace(responses=responses)
    env = wrappers.wrap_language_env(
        SensoryEnv(),
        state=lambda observation: observation["text"],
        tool_name="choose",
        argument_name="value",
    )
    policy = OpenAIResponsesPolicy.from_env(
        env,
        client=client,
        model="test-model",
        reasoning={"effort": "medium", "summary": "auto"},
    )
    observation, _ = env.reset()

    output = policy(observation)

    assert responses.request is not None
    assert responses.request["reasoning"] == {
        "effort": "medium",
        "summary": "auto",
    }
    assert output.info["reasoning_effort"] == "medium"
    assert output.info["reasoning_summary"] == (
        "Compared the available actions and selected two."
    )
