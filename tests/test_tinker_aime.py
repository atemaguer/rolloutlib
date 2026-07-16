"""Opt-in Tinker parity test for the AIME environment pipeline.

Run with ``RUN_TINKER_INTEGRATION=1`` after installing ``tinker``,
``tinker-cookbook``, and ``datasets`` and configuring ``TINKER_API_KEY``.
The test samples each answer once through Tinker, then grades the same decoded
responses with both rolloutlib's AIME environment and Tinker Cookbook's
reference ``AIMEMessageEnv``. This isolates grading/pipeline parity from
sampling variance between two independent benchmark runs.
"""

from __future__ import annotations

import asyncio
import importlib
import os

import pytest


if os.getenv("RUN_TINKER_INTEGRATION") != "1":
    pytest.skip(
        "set RUN_TINKER_INTEGRATION=1 to run the paid Tinker integration test",
        allow_module_level=True,
    )

tinker = importlib.import_module("tinker")
datasets = importlib.import_module("datasets")
renderers = importlib.import_module("tinker_cookbook.renderers")
reference_aime = importlib.import_module("tinker_cookbook.eval.benchmarks.aime")

from rolloutlib.evals import Evaluation, run_benchmark  # noqa: E402
from rolloutlib.evals.benchmarks import (  # noqa: E402
    AIME_PROMPT_SUFFIX,
    AIMEEnv,
    aime,
)
from rolloutlib.graders import Score  # noqa: E402


def test_aime_scores_match_tinker_cookbook_reference() -> None:
    model_name = os.getenv("TINKER_MODEL_NAME", "Qwen/Qwen3.5-4B")
    model_path = os.getenv("TINKER_MODEL_PATH")
    renderer_name = os.getenv("TINKER_RENDERER", "qwen3_5")
    dataset_name = os.getenv("TINKER_AIME_DATASET", "MathArena/aime_2025")
    limit = int(os.getenv("TINKER_AIME_LIMIT", "30"))
    max_tokens = int(os.getenv("TINKER_MAX_TOKENS", "32768"))
    temperature = float(os.getenv("TINKER_TEMPERATURE", "0.6"))
    system_prompt = os.getenv(
        "TINKER_SYSTEM_PROMPT", r"Put your final answer in \boxed{}."
    )

    try:
        dataset_split = "test"
        rows = list(datasets.load_dataset(dataset_name, split=dataset_split))
    except ValueError as exc:
        if 'Unknown split "test"' not in str(exc):
            raise
        dataset_split = "train"
        rows = list(datasets.load_dataset(dataset_name, split=dataset_split))
    rows = rows[:limit]
    benchmark = aime(
        rows,
        dataset=dataset_name,
        split=dataset_split,
        system_prompt=system_prompt,
        prompt_suffix=AIME_PROMPT_SUFFIX,
    )

    service_client = tinker.ServiceClient()
    if model_path:
        sampling_client = service_client.create_sampling_client(model_path=model_path)
    else:
        sampling_client = service_client.create_sampling_client(base_model=model_name)
    tokenizer = sampling_client.get_tokenizer()
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    sampling_params_type = getattr(
        tinker, "SamplingParams", tinker.types.SamplingParams
    )
    sampling_params = sampling_params_type(
        max_tokens=max_tokens,
        temperature=temperature,
        stop=renderer.get_stop_sequences(),
    )

    # Submit every request before waiting for results. This mirrors the
    # concurrency used by Tinker Cookbook and makes a full 30-problem run
    # practical while keeping rolloutlib's benchmark runner synchronous.
    benchmark_examples = list(benchmark.items)
    sampling_envs = [benchmark.make_env(example) for example in benchmark_examples]
    try:
        prompts = []
        example_ids = []
        for index, environment in enumerate(sampling_envs):
            observation, _ = environment.reset()
            prompts.append(renderer.build_generation_prompt(observation))
            example_ids.append(
                benchmark.item_id(benchmark_examples[index])
                if benchmark.item_id
                else str(index)
            )
        futures = [
            sampling_client.sample(
                prompt=prompt,
                sampling_params=sampling_params,
                num_samples=1,
            )
            for prompt in prompts
        ]
        sampled_results = [future.result() for future in futures]
    finally:
        for environment in sampling_envs:
            environment.close()

    responses: dict[str, str] = {}
    for example_id, sampled in zip(example_ids, sampled_results, strict=True):
        samples = getattr(sampled, "samples", None)
        if samples is None:
            # Older SDK releases called this field ``sequences``.
            samples = sampled.sequences
        message, _ = renderer.parse_response(samples[0].tokens)
        responses[example_id] = renderers.get_text_content(message)

    reference_scores: dict[str, float] = {}

    def evaluate(environment: AIMEEnv) -> Evaluation:
        observation, _ = environment.reset()
        del observation
        example_id = environment.example.example_id or ""
        response = responses[example_id]

        _, reward, terminated, truncated, info = environment.step(response)
        assert terminated or truncated

        expected = int(float(environment.example.answer.strip()))
        reference_env = reference_aime.AIMEMessageEnv(
            environment.example.question,
            expected,
            example_id=environment.example.example_id or "",
            system_prompt=system_prompt,
        )
        reference_result = asyncio.run(
            reference_env.step({"role": "assistant", "content": response})
        )
        reference_scores[example_id] = float(reference_result.reward)

        return Evaluation(
            score=Score.from_info(info, default=Score(reward)),
            truncated=truncated,
        )

    result = run_benchmark(benchmark, evaluate)
    rollout_scores = [
        record.score.value for record in result.records if record.score is not None
    ]

    expected_scores = [
        reference_scores[record.item_id or ""] for record in result.records
    ]
    print(
        f"AIME ({len(expected_scores)} examples): "
        f"score={result.score:.3f}, "
        f"correct={sum(expected_scores):.0f}/{len(expected_scores)}"
    )
    assert rollout_scores == pytest.approx(expected_scores)
    assert result.score == pytest.approx(sum(expected_scores) / len(expected_scores))
