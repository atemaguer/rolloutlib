"""Compare rolloutlib group collectors with Tinker Cookbook rollout patterns.

The default benchmark is deterministic and uses controlled policy latency:

    uv run python benchmarks/rollout_group_throughput.py

Add ``--live-tinker`` to compare real Tinker requests. This requires the
``tinker`` and ``tinker-cookbook`` packages plus ``TINKER_API_KEY``.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import os
import statistics
import time
from collections.abc import Sequence
from typing import Any

import gymnasium as gym

from rolloutlib.envs import AsyncEnv
from rolloutlib.rollouts import (
    PolicyOutput,
    abatched_rollout_group,
    rollout_group,
    vector_rollout_group,
)


class OneTurnEnv(gym.Env[int, int]):
    """Minimal single-turn environment for collector overhead measurements."""

    observation_space = gym.spaces.Discrete(1)
    action_space = gym.spaces.Discrete(1)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        super().reset(seed=seed)
        del options
        return 0, {}

    def step(self, action: int) -> tuple[int, float, bool, bool, dict[str, Any]]:
        return 0, float(action), True, False, {}


class AsyncOneTurnEnv(AsyncEnv[int, int]):
    """Async form of :class:`OneTurnEnv`."""

    observation_space = gym.spaces.Discrete(1)
    action_space = gym.spaces.Discrete(1)

    async def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        await super().reset(seed=seed, options=options)
        return 0, {}

    async def step(
        self,
        action: int,
    ) -> tuple[int, float, bool, bool, dict[str, Any]]:
        return 0, float(action), True, False, {}


def _median_seconds(function: Any, repeats: int) -> float:
    measurements = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        measurements.append(time.perf_counter() - start)
    return statistics.median(measurements)


def _measure_once(function: Any) -> float:
    start = time.perf_counter()
    function()
    return time.perf_counter() - start


def controlled_benchmark(
    *,
    group_size: int,
    latency_seconds: float,
    repeats: int,
) -> dict[str, float]:
    """Measure collector scheduling with identical simulated sampling latency."""

    def scalar_policy(_: int) -> int:
        time.sleep(latency_seconds)
        return 0

    def batch_policy(observations: Sequence[int]) -> list[int]:
        time.sleep(latency_seconds)
        return [0] * len(observations)

    async def async_batch_policy(observations: Sequence[int]) -> list[int]:
        await asyncio.sleep(latency_seconds)
        return [0] * len(observations)

    def run_sequential() -> None:
        rollout_group(
            None,
            lambda _: OneTurnEnv(),
            scalar_policy,
            num_rollouts=group_size,
        )

    def run_vector() -> None:
        vector_rollout_group(
            None,
            lambda _: OneTurnEnv(),
            batch_policy,
            num_rollouts=group_size,
        )

    def run_active() -> None:
        asyncio.run(
            abatched_rollout_group(
                None,
                lambda _: AsyncOneTurnEnv(),
                async_batch_policy,
                num_rollouts=group_size,
            )
        )

    try:
        cookbook_modules: tuple[Any, Any, Any, Any] | None = (
            importlib.import_module("tinker"),
            importlib.import_module("tinker_cookbook.completers"),
            importlib.import_module("tinker_cookbook.rl.rollouts"),
            importlib.import_module("tinker_cookbook.rl.types"),
        )
    except ImportError:
        cookbook_modules = None

    def run_cookbook() -> None:
        if cookbook_modules is None:
            raise RuntimeError("tinker-cookbook is required for this comparison")
        (
            tinker_module,
            completers_module,
            rollouts_module,
            types_module,
        ) = cookbook_modules

        class CookbookEnv(types_module.Env):
            async def initial_observation(self) -> tuple[Any, list[int]]:
                return tinker_module.ModelInput.from_ints([1]), []

            async def step(
                self,
                action: list[int],
                *,
                extra: Any = None,
            ) -> Any:
                del action, extra
                return types_module.StepResult(
                    reward=0.0,
                    episode_done=True,
                    next_observation=tinker_module.ModelInput.from_ints([]),
                    next_stop_condition=[],
                )

        class Builder(types_module.EnvGroupBuilder):
            async def make_envs(self) -> list[CookbookEnv]:
                return [CookbookEnv() for _ in range(group_size)]

        class Completer(completers_module.TokenCompleter):
            async def __call__(
                self,
                observation: Any,
                stop: Any,
                *,
                max_tokens: int | None = None,
            ) -> Any:
                del observation, stop, max_tokens
                await asyncio.sleep(latency_seconds)
                return completers_module.TokensWithLogprobs(
                    tokens=[1],
                    maybe_logprobs=[0.0],
                )

        asyncio.run(rollouts_module.do_group_rollout(Builder(), Completer()))

    measurements = {
        "rolloutlib_sequential_seconds": _median_seconds(run_sequential, repeats),
        "rolloutlib_vector_seconds": _median_seconds(run_vector, repeats),
        "rolloutlib_active_batch_seconds": _median_seconds(run_active, repeats),
    }
    if cookbook_modules is not None:
        measurements["tinker_cookbook_seconds"] = _median_seconds(
            run_cookbook,
            repeats,
        )
    vector = measurements["rolloutlib_vector_seconds"]
    measurements["vector_speedup_over_sequential"] = (
        measurements["rolloutlib_sequential_seconds"] / vector
    )
    if "tinker_cookbook_seconds" in measurements:
        measurements["vector_to_cookbook_ratio"] = (
            vector / measurements["tinker_cookbook_seconds"]
        )
        measurements["active_batch_to_cookbook_ratio"] = (
            measurements["rolloutlib_active_batch_seconds"]
            / measurements["tinker_cookbook_seconds"]
        )
    return measurements


def live_tinker_benchmark(
    *,
    group_size: int,
    max_tokens: int,
    repeats: int,
) -> dict[str, float | str]:
    """Compare grouped and concurrent real Tinker sampling calls."""
    import tinker
    from tinker_cookbook.completers import TinkerTokenCompleter
    from tinker_cookbook.rl.rollouts import do_group_rollout
    from tinker_cookbook.rl.types import Env, EnvGroupBuilder, StepResult

    model_name = os.getenv("TINKER_MODEL_NAME", "Qwen/Qwen3.5-4B")
    model_path = os.getenv("TINKER_MODEL_PATH")
    service_client = tinker.ServiceClient()
    if model_path:
        sampling_client = service_client.create_sampling_client(model_path=model_path)
    else:
        sampling_client = service_client.create_sampling_client(base_model=model_name)
    tokenizer = sampling_client.get_tokenizer()
    prompt = tinker.ModelInput.from_ints(
        tokenizer.encode("Reply with exactly one word: ready")
    )
    sampling_params_type = getattr(tinker, "SamplingParams", tinker.types.SamplingParams)
    sampling_params = sampling_params_type(
        max_tokens=max_tokens,
        temperature=0.0,
    )

    class CookbookEnv(Env):
        async def initial_observation(self) -> tuple[Any, list[int]]:
            return prompt, []

        async def step(
            self,
            action: list[int],
            *,
            extra: Any = None,
        ) -> StepResult:
            del action, extra
            return StepResult(
                reward=0.0,
                episode_done=True,
                next_observation=tinker.ModelInput.from_ints([]),
                next_stop_condition=[],
            )

    class Builder(EnvGroupBuilder):
        async def make_envs(self) -> list[CookbookEnv]:
            return [CookbookEnv() for _ in range(group_size)]

    cookbook_policy = TinkerTokenCompleter(
        sampling_client,
        max_tokens=max_tokens,
        temperature=0.0,
    )

    def grouped_sample() -> Any:
        return sampling_client.sample(
            prompt,
            num_samples=group_size,
            sampling_params=sampling_params,
        ).result()

    def vector_policy(observations: Sequence[int]) -> list[PolicyOutput[int]]:
        response = grouped_sample()
        sequences = getattr(response, "sequences", None) or getattr(
            response,
            "samples",
        )
        return [
            PolicyOutput(
                action=0,
                tokens=sequence.tokens,
                logprobs=getattr(sequence, "logprobs", None),
                stop_reason=str(getattr(sequence, "stop_reason", "")),
            )
            for sequence in sequences
        ]

    def run_vector() -> None:
        vector_rollout_group(
            None,
            lambda _: OneTurnEnv(),
            vector_policy,
            num_rollouts=group_size,
        )

    async def active_policy(observations: Sequence[int]) -> list[PolicyOutput[int]]:
        return await asyncio.to_thread(vector_policy, observations)

    def run_active() -> None:
        asyncio.run(
            abatched_rollout_group(
                None,
                lambda _: AsyncOneTurnEnv(),
                active_policy,
                num_rollouts=group_size,
            )
        )

    def run_cookbook() -> None:
        asyncio.run(do_group_rollout(Builder(), cookbook_policy))

    sampling_client.sample(
        prompt,
        num_samples=1,
        sampling_params=sampling_params,
    ).result()

    runners = {
        "raw_grouped_sample_seconds": grouped_sample,
        "rolloutlib_vector_seconds": run_vector,
        "rolloutlib_active_batch_seconds": run_active,
        "tinker_cookbook_concurrent_seconds": run_cookbook,
    }
    measurements_by_name: dict[str, list[float]] = {
        name: [] for name in runners
    }
    names = list(runners)
    for repeat in range(repeats):
        order = names[repeat % len(names) :] + names[: repeat % len(names)]
        for name in order:
            measurements_by_name[name].append(_measure_once(runners[name]))

    measurements = {
        "model": model_path or model_name,
        **{
            name: statistics.median(values)
            for name, values in measurements_by_name.items()
        },
    }
    measurements["rolloutlib_vector_over_raw_ratio"] = (
        measurements["rolloutlib_vector_seconds"]
        / measurements["raw_grouped_sample_seconds"]
    )
    measurements["rolloutlib_vector_to_cookbook_ratio"] = (
        measurements["rolloutlib_vector_seconds"]
        / measurements["tinker_cookbook_concurrent_seconds"]
    )
    measurements["rolloutlib_active_to_cookbook_ratio"] = (
        measurements["rolloutlib_active_batch_seconds"]
        / measurements["tinker_cookbook_concurrent_seconds"]
    )
    return measurements


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--latency-ms", type=float, default=25.0)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--live-tinker", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=8)
    args = parser.parse_args()

    results: dict[str, Any] = {
        "configuration": {
            "group_size": args.group_size,
            "latency_ms": args.latency_ms,
            "repeats": args.repeats,
        },
        "controlled": controlled_benchmark(
            group_size=args.group_size,
            latency_seconds=args.latency_ms / 1000.0,
            repeats=args.repeats,
        ),
    }
    if args.live_tinker:
        results["live_tinker"] = live_tinker_benchmark(
            group_size=args.group_size,
            max_tokens=args.max_tokens,
            repeats=args.repeats,
        )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
