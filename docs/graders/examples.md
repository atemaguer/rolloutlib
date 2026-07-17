# Representative grader examples

These examples show how the grader abstractions fit common post-training and
evaluation workloads. They use application-defined inputs so the same patterns
extend beyond text responses.

## Programmatic short-answer grading

This complete example combines exact matching, format validation, and a small
brevity bonus.

```python
from pydantic import BaseModel

from rolloutlib.graders import RewardGrader, Score
from rolloutlib.spaces import PydanticSpace


class AnswerInput(BaseModel):
    question: str
    response: str
    reference_answer: str


answer_space = PydanticSpace(AnswerInput)


def exact_match(item: AnswerInput) -> Score:
    candidate = " ".join(item.response.casefold().split())
    reference = " ".join(item.reference_answer.casefold().split())
    correct = candidate == reference
    return Score(
        float(correct),
        feedback="The normalized response matches the reference."
        if correct
        else "The normalized response does not match the reference.",
    )


def requested_format(item: AnswerInput) -> float:
    return float(item.response.startswith("Answer:"))


def concise(item: AnswerInput) -> float:
    return float(len(item.response.split()) <= 20)


grader = RewardGrader(
    {
        "exact_match": exact_match,
        "requested_format": requested_format,
        "concise": concise,
    },
    input_space=answer_space,
    weights={
        "exact_match": 1.0,
        "requested_format": 0.1,
        "concise": 0.05,
    },
    metadata={"grader_version": "1"},
)

item = AnswerInput(
    question="What is 6 × 7?",
    response="Answer: 42",
    reference_answer="Answer: 42",
)
score = grader.grade(item)

assert round(score.value, 2) == 1.15
assert score.component_values == {
    "exact_match": 1.0,
    "requested_format": 1.0,
    "concise": 1.0,
}
```

The final reward is a weighted sum. Each underlying signal remains available
for metrics and debugging.

## A classroom-style rubric

This example defines discrete levels for correctness and continuous scoring for
clarity.

```python
from rolloutlib import Criterion, Level, Rubric, spaces
from rolloutlib.graders import RubricGrader, Score

rubric = Rubric(
    id="math-explanation",
    version="1",
    title="Mathematical explanation",
    instructions="Assess the submitted answer without adding missing work.",
    criteria=(
        Criterion(
            id="correctness",
            title="Correctness",
            description="The conclusion and mathematical reasoning are correct.",
            weight=3.0,
            levels=(
                Level(
                    id="complete",
                    description="The conclusion and all reasoning are correct.",
                    score=1.0,
                ),
                Level(
                    id="partial",
                    description="The conclusion is correct with a reasoning gap.",
                    score=0.5,
                ),
                Level(
                    id="incorrect",
                    description="The conclusion is incorrect.",
                    score=0.0,
                ),
            ),
        ),
        Criterion(
            id="clarity",
            title="Clarity",
            description="The explanation is direct and easy to follow.",
            weight=1.0,
        ),
    ),
)


def local_judge(response: str, rubric: Rubric):
    has_correct_result = "42" in response
    explains_multiplication = "6" in response and "7" in response
    correctness = 1.0 if has_correct_result and explains_multiplication else 0.5
    clarity = 1.0 if len(response.split()) <= 30 else 0.5
    return {
        "correctness": Score(
            correctness,
            feedback="The answer states 42 and connects it to 6 × 7.",
        ),
        "clarity": Score(
            clarity,
            feedback="The explanation is concise.",
        ),
    }


grader = RubricGrader(
    rubric,
    local_judge,
    input_space=spaces.text.text(min_length=1),
)

score = grader.grade("6 multiplied by 7 equals 42.")

assert score.value == 1.0
assert score.metadata["rubric_id"] == "math-explanation"
assert score.metadata["rubric_version"] == "1"
assert score.metadata["rubric_fingerprint"] == rubric.fingerprint
```

The judge happens to be deterministic, demonstrating that rubric graders are
defined by their judgment policy rather than by a required model provider.

## An asynchronous LLM rubric judge

The following integration is provider-neutral. The application owns the model
client, prompt, and response parsing; Rolloutlib owns validation, criterion
matching, aggregation, and score structure.

```python
import json
from typing import Protocol

from pydantic import BaseModel, Field

from rolloutlib import Criterion, Rubric
from rolloutlib.graders import AsyncRubricGrader, Score
from rolloutlib.spaces import PydanticSpace


class GradingInput(BaseModel):
    prompt: str
    response: str
    reference_answer: str


class CriterionResult(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    feedback: str


class JudgeResponse(BaseModel):
    criteria: dict[str, CriterionResult]


class ModelClient(Protocol):
    model_name: str

    async def generate_json(self, request: str) -> str: ...


rubric = Rubric(
    id="answer-quality",
    version="2",
    criteria=(
        Criterion(
            id="correctness",
            description="The response agrees with the reference answer.",
            weight=3.0,
        ),
        Criterion(
            id="reasoning",
            description="The response explains how it reaches its conclusion.",
            weight=2.0,
        ),
    ),
)

grading_space = PydanticSpace(GradingInput)


def render_request(item: GradingInput, rubric: Rubric) -> str:
    return json.dumps(
        {
            "task": item.model_dump(),
            "rubric": rubric.model_dump(mode="json"),
            "response_format": {
                "criteria": {
                    criterion.id: {
                        "score": "number from 0 through 1",
                        "feedback": "short explanation",
                    }
                    for criterion in rubric.criteria
                }
            },
        }
    )


def make_grader(client: ModelClient) -> AsyncRubricGrader[GradingInput]:
    async def judge(item: GradingInput, rubric: Rubric):
        request = render_request(item, rubric)
        raw_response = await client.generate_json(request)
        response = JudgeResponse.model_validate_json(raw_response)
        return {
            criterion_id: Score(
                result.score,
                metadata={"judge_model": client.model_name},
                feedback=result.feedback,
            )
            for criterion_id, result in response.criteria.items()
        }

    return AsyncRubricGrader(
        rubric,
        judge,
        input_space=grading_space,
        metadata={
            "grader": "answer-quality",
            "prompt_version": "4",
        },
    )
```

`AsyncRubricGrader` verifies that the parsed mapping contains exactly
`correctness` and `reasoning`. A provider response that omits or invents a
criterion fails explicitly.

In a production judge, configure the provider to emit a structured response
matching `JudgeResponse`, calibrate the scores against human labels, and record
provider request identifiers in metadata.

## A hybrid LLM and deterministic grader

This pattern uses a rubric judge for qualitative quality and a reward grader
for objective verification.

```python
from rolloutlib.graders import (
    AsyncCompositeGrader,
    AsyncRubricGrader,
    RewardGrader,
    Score,
)


def exact_match(item: GradingInput) -> Score:
    correct = item.response.strip() == item.reference_answer.strip()
    return Score(float(correct), feedback="Compared with the reference answer.")


def non_empty(item: GradingInput) -> float:
    return float(bool(item.response.strip()))


verification = RewardGrader(
    {
        "exact_match": exact_match,
        "non_empty": non_empty,
    },
    input_space=grading_space,
    weights={
        "exact_match": 1.0,
        "non_empty": 0.0,
    },
)


def make_hybrid_grader(client: ModelClient):
    quality: AsyncRubricGrader[GradingInput] = make_grader(client)
    return AsyncCompositeGrader(
        {
            "quality": quality,
            "verification": verification,
        },
        input_space=grading_space,
        weights={
            "quality": 0.7,
            "verification": 0.3,
        },
        metadata={"reward_policy": "hybrid-v1"},
    )
```

`non_empty` has zero weight, so it is recorded without affecting the
verification scalar. The composite preserves the complete hierarchy:

```text
score
├── quality
│   ├── correctness
│   └── reasoning
└── verification
    ├── exact_match
    └── non_empty
```

If exact correctness must veto the qualitative score, use a custom composite
aggregate:

```python
def require_exact_match(scores) -> float:
    exact = scores["verification"].components["exact_match"].value
    if exact < 1.0:
        return 0.0
    return scores["quality"].value
```

## Grading an environment action

`GradingWrapper` adds grading to an existing Gymnasium environment. This
complete example grades the terminal answer and retains the structured score in
`info`.

```python
import gymnasium as gym

from rolloutlib import wrappers
from rolloutlib.graders import RewardGrader, Score


class QuestionEnv(gym.Env[str, str]):
    action_space = gym.spaces.Text(min_length=1, max_length=100)
    observation_space = gym.spaces.Text(min_length=1, max_length=100)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return "What is 6 × 7?", {}

    def step(self, action):
        return "done", 0.0, True, False, {}


grader = RewardGrader(
    {
        "correct": lambda answer: Score(
            float(answer.strip() == "42"),
            feedback="Checked against the task answer.",
        )
    },
    input_space=QuestionEnv.action_space,
)

environment = wrappers.GradingWrapper(
    QuestionEnv(),
    grader=grader,
    make_input=lambda env, action: action,
)

observation, reset_info = environment.reset()
observation, reward, terminated, truncated, info = environment.step("42")

assert reward == 1.0
assert terminated is True
assert Score.from_info(info) == grader.grade("42")
```

For an async environment or grader, use `AsyncGradingWrapper`. It accepts both
sync and async graders.

## Combining an existing environment reward

The default wrapper replaces the inner reward. Supply `combine_reward` when the
inner signal should remain:

```python
environment = wrappers.GradingWrapper(
    QuestionEnv(),
    grader=grader,
    make_input=lambda env, action: action,
    combine_reward=lambda environment_reward, score: (
        0.25 * environment_reward + 0.75 * score.value
    ),
)
```

The full grader score is still stored in `info`, while the environment returns
the combined scalar.

## Grading a tool-use trace

Grader inputs need not be strings. A structured trace can expose task outcome,
tool selection, and argument validity to separate reward functions.

```python
from pydantic import BaseModel

from rolloutlib.graders import RewardGrader
from rolloutlib.spaces import PydanticSpace


class ToolTrace(BaseModel):
    requested_tool: str
    called_tool: str | None
    arguments_valid: bool
    task_completed: bool
    tool_calls: int


tool_trace_space = PydanticSpace(ToolTrace)

tool_grader = RewardGrader(
    {
        "tool_selection": lambda trace: float(
            trace.called_tool == trace.requested_tool
        ),
        "argument_validity": lambda trace: float(trace.arguments_valid),
        "task_success": lambda trace: float(trace.task_completed),
        "efficiency": lambda trace: 1.0 / max(trace.tool_calls, 1),
    },
    input_space=tool_trace_space,
    weights={
        "tool_selection": 0.25,
        "argument_validity": 0.25,
        "task_success": 1.0,
        "efficiency": 0.1,
    },
)

trace = ToolTrace(
    requested_tool="search",
    called_tool="search",
    arguments_valid=True,
    task_completed=True,
    tool_calls=2,
)

score = tool_grader.grade(trace)

assert score.component_values == {
    "tool_selection": 1.0,
    "argument_validity": 1.0,
    "task_success": 1.0,
    "efficiency": 0.5,
}
```

For a full trajectory, define a space around the trajectory representation used
by the application and follow the same pattern.

## Per-item rubrics

When dataset items carry different rubrics, bind each rubric while constructing
the task runtime:

```python
class DatasetItem(BaseModel):
    prompt: str
    reference_answer: str
    rubric: Rubric


def make_grader(item: DatasetItem, client: ModelClient):
    async def judge(input: GradingInput, rubric: Rubric):
        raw = await client.generate_json(render_request(input, rubric))
        parsed = JudgeResponse.model_validate_json(raw)
        return {
            name: Score(result.score, feedback=result.feedback)
            for name, result in parsed.criteria.items()
        }

    return AsyncRubricGrader(
        item.rubric,
        judge,
        input_space=grading_space,
    )
```

The caller still uses `await grader.grade(input)`; rubric selection is part of
constructing the configured grader, not part of every grade operation.

## A custom grader protocol

The standard families cover rubric judgment, named reward functions, and
composition. A specialized protocol can implement the base contract directly:

```python
import gymnasium as gym

from rolloutlib import Grader, Score, spaces


class PairwisePreferenceGrader(Grader[tuple[str, str]]):
    input_space = gym.spaces.Tuple(
        (
            spaces.text.text(min_length=1),
            spaces.text.text(min_length=1),
        )
    )

    def _grade(self, input: tuple[str, str]) -> Score:
        first, second = input
        first_is_shorter = len(first) < len(second)
        return Score(
            float(first_is_shorter),
            metadata={"preferred": "first" if first_is_shorter else "second"},
        )
```

Custom subclasses should still return structured scores and let the public
`grade` method enforce the declared input space.

## Related documentation

- [Grader concepts](../concepts/graders.md)
- [Rubrics and rubric graders](rubrics.md)
- [Reward graders](reward-graders.md)
- [Composite graders](composite-graders.md)
- [Scores and aggregation](scores-and-aggregation.md)
