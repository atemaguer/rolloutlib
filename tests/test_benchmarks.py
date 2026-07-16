from __future__ import annotations

from typing import Any, cast

import pytest

from rolloutlib.evals import Evaluation, run_benchmark
from rolloutlib.evals.benchmarks import (
    AIMEEnv,
    GSM8KEnv,
    aime,
    extract_aime_answer,
    extract_gsm8k_answer,
    gsm8k,
)
from rolloutlib.graders import Score


def test_math_answer_extractors_handle_common_formats() -> None:
    assert extract_gsm8k_answer("Work ...\n#### 1,200") == "1200"
    assert extract_gsm8k_answer(r"The result is \boxed{42}.") == "42"
    assert extract_aime_answer(r"Therefore, \boxed{007}.") == "007"
    assert extract_aime_answer("The final answer is 314") == "314"


def test_gsm8k_environment_is_single_turn_and_scores_step() -> None:
    benchmark = gsm8k(
        [
            {
                "question": "What is 6 times 7?",
                "answer": "The answer is 42.\n#### 42",
            }
        ]
    )
    environment = benchmark.make_env(benchmark.items[0])  # type: ignore[index]
    assert isinstance(environment, GSM8KEnv)

    observation, reset_info = environment.reset()
    assert observation[0]["content"] == "What is 6 times 7?"
    assert reset_info["example_id"].startswith("gsm8k:")
    _, reward, terminated, truncated, info = environment.step("The answer is 42.")

    assert reward == 1.0
    assert terminated is True
    assert truncated is False
    score = Score.from_info(info)
    assert score is not None
    assert score.component_values == {"correct": 1.0}
    assert environment.observation_space.contains([{"role": "assistant", "content": "The answer is 42."}])
    environment.close()


def test_math_environment_observation_space_supports_system_prompt() -> None:
    benchmark = gsm8k(
        [{"question": "What is 6 times 7?", "answer": "#### 42"}],
        system_prompt="Solve carefully.",
    )
    environment = benchmark.make_env(benchmark.items[0])
    observation, _ = environment.reset()
    assert observation in environment.observation_space
    _, _, _, _, _ = environment.step("42")
    environment.close()


def test_benchmark_runner_passes_fresh_envs_to_user_callback() -> None:
    benchmark = aime(
        [
            {"problem": "AIME problem one", "answer": "42", "id": 1},
            {"problem": "AIME problem two", "answer": "17", "id": 2},
        ]
    )
    seen: list[str] = []

    def evaluate(environment: Any) -> Evaluation:
        assert isinstance(environment, AIMEEnv)
        observation, _ = environment.reset()
        seen.append(cast(str, observation[0]["content"]))
        _, reward, terminated, truncated, info = environment.step(
            "I conclude the answer is 42." if len(seen) == 1 else "17"
        )
        assert terminated and not truncated
        return Evaluation(score=Score.from_info(info, default=Score(reward)))

    result = run_benchmark(benchmark, evaluate=evaluate)

    assert result.score == pytest.approx(1.0)
    assert result.num_examples == 2
    assert [record.item_id for record in result.records] == ["1", "2"]
    assert seen == [
        "AIME problem one\n\nThis is an AIME problem. The answer is an integer from 000 to 999. "
        "Show your work step by step, then put your final answer in \\boxed{}.",
        "AIME problem two\n\nThis is an AIME problem. The answer is an integer from 000 to 999. "
        "Show your work step by step, then put your final answer in \\boxed{}.",
    ]
