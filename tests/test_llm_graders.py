from __future__ import annotations

import asyncio

import gymnasium as gym

from rolloutlib.graders import (
    AsyncLLMGrader,
    Criterion,
    LLMGrader,
    Rubric,
    Score,
)
from rolloutlib.types import Chat


def rubric() -> Rubric:
    return Rubric(
        id="correctness",
        criteria=(Criterion(id="correct", description="The answer is correct."),),
    )


def render(answer: str, value: Rubric | None) -> Chat:
    assert value is not None
    return [
        {
            "role": "user",
            "content": f"{value.criteria[0].description}\nAnswer: {answer}",
        }
    ]


def parse(response: str, value: Rubric | None) -> Score:
    del value
    return Score(float(response), feedback="Parsed judge score.")


def test_llm_grader_adapts_a_synchronous_sampler() -> None:
    grader = LLMGrader[str](
        input_space=gym.spaces.Text(max_length=100),
        sample=lambda messages: "1" if "42" in messages[0]["content"] else "0",
        render=render,
        parse=parse,
        metadata={"model": "test-judge"},
    )

    score = grader.grade("42", rubric=rubric())

    assert score.value == 1.0
    assert score.feedback == "Parsed judge score."
    assert score.metadata["model"] == "test-judge"
    assert score.metadata["rubric_id"] == "correctness"


def test_async_llm_grader_adapts_an_asynchronous_sampler() -> None:
    async def run() -> None:
        async def sample(messages: Chat) -> str:
            await asyncio.sleep(0)
            return "1"

        grader = AsyncLLMGrader[str](
            input_space=gym.spaces.Text(max_length=100),
            sample=sample,
            render=render,
            parse=parse,
        )

        assert (await grader.grade("42", rubric=rubric())).value == 1.0

    asyncio.run(run())
