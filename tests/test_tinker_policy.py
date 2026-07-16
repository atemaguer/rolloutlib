"""Opt-in smoke test for a Policy implementation backed by Tinker.

This intentionally lives in tests rather than the rolloutlib package. Tinker
is an optional backend; the public contract is :class:`PolicyOutput`, not a
Tinker-specific adapter.

Run with ``RUN_TINKER_INTEGRATION=1`` after installing ``tinker`` and
``tinker-cookbook`` and configuring Tinker credentials.
"""

from __future__ import annotations

import importlib
import os
from collections.abc import Callable, Sequence
from typing import Any

import gymnasium as gym
import pytest


if os.getenv("RUN_TINKER_INTEGRATION") != "1":
    pytest.skip(
        "set RUN_TINKER_INTEGRATION=1 to run the paid Tinker integration test",
        allow_module_level=True,
    )

tinker = importlib.import_module("tinker")
renderers = importlib.import_module("tinker_cookbook.renderers")

from rolloutlib import Policy, PolicyOutput, rollout  # noqa: E402
from rolloutlib import spaces as rollout_spaces  # noqa: E402


class TinkerPolicy(Policy[Any, Any]):
    """Small test-only policy showing the intended Tinker composition."""

    def __init__(
        self,
        sampling_client: Any,
        renderer: Any,
        sampling_params: Any,
        *,
        decode: Callable[[Any], Any],
    ) -> None:
        self.sampling_client = sampling_client
        self.renderer = renderer
        self.sampling_params = sampling_params
        self.decode = decode

    def __call__(self, observation: Any) -> PolicyOutput[Any]:
        model_input = self.renderer.build_generation_prompt(observation)
        future = self.sampling_client.sample(
            model_input,
            num_samples=1,
            sampling_params=self.sampling_params,
        )
        result = future.result()
        sequences = getattr(result, "sequences", None) or getattr(result, "samples")
        sequence = sequences[0]
        tokens = tuple(int(token) for token in sequence.tokens)
        message, termination = self.renderer.parse_response(tokens)
        logprobs = getattr(sequence, "logprobs", None)
        if logprobs is None:
            logprobs = getattr(sequence, "maybe_logprobs", None)
        return PolicyOutput(
            action=self.decode(message),
            info={"message": message, "termination": termination},
            tokens=tokens,
            logprobs=logprobs,
            stop_reason=str(getattr(sequence, "stop_reason", "")),
        )


class OneStepChatEnv(gym.Env[Any, str]):
    """Minimal environment used to verify policy/rollout composition."""

    action_space = gym.spaces.Text(min_length=0, max_length=4096)
    observation_space = rollout_spaces.messages.chat(min_length=1)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        super().reset(seed=seed)
        del options
        return [{"role": "user", "content": "What is 2 + 2?"}], {}

    def step(self, action: str) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        return [], 1.0, True, False, {"response": action}


def test_tinker_policy_returns_policy_output() -> None:
    model_name = os.getenv("TINKER_MODEL_NAME", "Qwen/Qwen3.5-4B")
    model_path = os.getenv("TINKER_MODEL_PATH")
    renderer_name = os.getenv("TINKER_RENDERER", "qwen3_5")
    max_tokens = int(os.getenv("TINKER_MAX_TOKENS", "128"))
    temperature = float(os.getenv("TINKER_TEMPERATURE", "0.0"))

    service_client = tinker.ServiceClient()
    if model_path:
        sampling_client = service_client.create_sampling_client(model_path=model_path)
    else:
        sampling_client = service_client.create_sampling_client(base_model=model_name)
    tokenizer = sampling_client.get_tokenizer()
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    sampling_params_type = getattr(tinker, "SamplingParams", tinker.types.SamplingParams)
    sampling_params = sampling_params_type(
        max_tokens=max_tokens,
        temperature=temperature,
        stop=renderer.get_stop_sequences(),
    )
    get_text_content = renderers.get_text_content
    policy = TinkerPolicy(
        sampling_client,
        renderer,
        sampling_params,
        decode=get_text_content,
    )

    trajectory = rollout(OneStepChatEnv(), policy)

    assert trajectory.complete
    assert trajectory.total_reward == 1.0
    assert len(trajectory.steps) == 1
    step = trajectory.steps[0]
    assert step.action
    assert step.policy_tokens
    assert isinstance(step.policy_tokens, Sequence)
    assert step.policy_info["message"]["role"] == "assistant"
    assert step.info["response"] == step.action
