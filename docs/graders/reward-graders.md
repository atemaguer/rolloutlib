# Reward graders

A `RewardGrader` turns separately named programmatic reward functions into one
structured grading result. It is the standard choice for deterministic checks,
test execution, validators, heuristics, and other signals whose behavior lives
in code.

## The reward function contract

A synchronous reward function receives exactly one value—the grader input—and
returns a scalar or `Score`:

```python
def reward(input) -> float | Score:
    ...
```

An async reward function may return that value directly or await it:

```python
async def reward(input) -> float | Score:
    ...
```

The input contains all required context. A reward function does not receive a
rubric or grader instance.

## A minimal reward grader

```python
from rolloutlib import spaces
from rolloutlib.graders import RewardGrader


def exact_answer(response: str) -> float:
    return float(response.strip() == "42")


grader = RewardGrader(
    {"exact_answer": exact_answer},
    input_space=spaces.text.text(min_length=1),
)

score = grader.grade("42")

assert score.value == 1.0
assert score.component_values == {"exact_answer": 1.0}
```

Even a single function becomes a named component. Consumers can therefore
distinguish the component signal from the final aggregation.

Constructor options:

| Argument | Meaning |
| --- | --- |
| `rewards` | Mapping from stable component names to reward functions |
| `input_space` | Space describing accepted grader inputs |
| `weights` | Optional non-negative component weights |
| `aggregate` | Optional function producing the final scalar |
| `metadata` | Metadata attached to every top-level result |

## Rich grader inputs

Reward functions often need references or task state. Put that context in the
grader input:

```python
from pydantic import BaseModel

from rolloutlib.spaces import PydanticSpace


class AnswerInput(BaseModel):
    question: str
    response: str
    reference_answer: str


answer_space = PydanticSpace(AnswerInput)
```

Functions can then read only the fields they need:

```python
def exact_match(input: AnswerInput) -> float:
    return float(input.response.strip() == input.reference_answer.strip())


def concise(input: AnswerInput) -> float:
    return float(len(input.response.split()) <= 100)
```

## Returning `Score`

Return a scalar when only the value matters. Return `Score` to preserve
feedback, metadata, or subcomponents:

```python
from rolloutlib.graders import Score


def keyword_coverage(input: AnswerInput) -> Score:
    required = {"photosynthesis", "chlorophyll", "sunlight"}
    present = {
        keyword
        for keyword in required
        if keyword in input.response.casefold()
    }
    value = len(present) / len(required)
    return Score(
        value,
        {
            keyword: float(keyword in present)
            for keyword in sorted(required)
        },
        metadata={"required_keywords": sorted(required)},
        feedback=f"Found {len(present)} of {len(required)} required concepts.",
    )
```

The component result is retained unchanged under the top-level grader score.

## Multiple reward functions

Pass a mapping from stable names to functions:

```python
grader = RewardGrader(
    {
        "exact_match": exact_match,
        "concise": concise,
        "keyword_coverage": keyword_coverage,
    },
    input_space=answer_space,
)
```

Names must be non-empty. At least one reward function is required. Mapping
insertion order is preserved in the resulting components.

Keep names stable because they become part of stored score records, metrics,
and custom aggregation policies.

## Weights and default aggregation

The default aggregate is a weighted sum:

```text
value = Σ weight[name] × component[name].value
```

Every unspecified weight defaults to `1.0`:

```python
grader = RewardGrader(
    {
        "correctness": exact_match,
        "format": valid_format,
    },
    input_space=answer_space,
    weights={
        "correctness": 1.0,
        "format": 0.1,
    },
)
```

Weights must be finite and non-negative. Zero disables a component's
contribution while retaining its diagnostic result. At least one weight must be
positive, and weights cannot reference unknown component names.

Weighted sum is useful for additive shaping signals and bonus rewards. It does
not normalize the result to `[0, 1]`; a grader with three unit-valued components
and unit weights returns `3.0`.

Use a custom aggregate when a normalized or gated result is required.

## Custom aggregation

A custom reward aggregator receives the completed mapping of component names to
`Score` objects and returns the final scalar:

```python
def normalized_mean(scores) -> float:
    return sum(score.value for score in scores.values()) / len(scores)


grader = RewardGrader(
    {
        "correctness": exact_match,
        "format": valid_format,
    },
    input_space=answer_space,
    aggregate=normalized_mean,
)
```

The aggregator always receives `Score` values, even when the original functions
returned floats.

Gating policies are also straightforward:

```python
def tests_gate_quality(scores) -> float:
    if scores["tests"].value < 1.0:
        return 0.0
    return scores["quality"].value
```

When `aggregate` is supplied, it owns the final value. The `weights` mapping
remains grader configuration but is not automatically applied inside the custom
function.

## Asynchronous reward functions

Use `RewardGrader` when any function performs asynchronous I/O:

```python
from rolloutlib.graders import RewardGrader


async def remote_safety_check(input: AnswerInput) -> float:
    result = await safety_client.check(input.response)
    return float(result.allowed)


async def groundedness_check(input: AnswerInput) -> Score:
    result = await retrieval_client.verify(
        question=input.question,
        response=input.response,
    )
    return Score(
        result.score,
        feedback=result.explanation,
    )


grader = RewardGrader(
    {
        "safety": remote_safety_check,
        "groundedness": groundedness_check,
        "exact_match": exact_match,
    },
    input_space=answer_space,
)

score = await grader.grade(item)
```

The same `RewardGrader` accepts both sync and async reward functions.
Independent awaitable functions are scheduled concurrently, while an all-sync
configuration returns `Score` directly. The order of `score.components` still
matches the configured mapping.

## Common patterns

### Exact or normalized matching

```python
def normalized_exact_match(input: AnswerInput) -> float:
    candidate = " ".join(input.response.casefold().split())
    reference = " ".join(input.reference_answer.casefold().split())
    return float(candidate == reference)
```

Keep normalization explicit. Different tasks may reasonably disagree about
case, punctuation, units, or whitespace.

### Format validation

```python
import json


def valid_json(input: AnswerInput) -> Score:
    try:
        json.loads(input.response)
    except json.JSONDecodeError as error:
        return Score(0.0, feedback=f"Invalid JSON: {error.msg}.")
    return Score(1.0, feedback="The response is valid JSON.")
```

### Test-based rewards

```python
class CodeInput(BaseModel):
    solution: str
    tests: tuple[str, ...]


def passes_tests(input: CodeInput) -> Score:
    result = sandbox.run_tests(input.solution, input.tests)
    return Score(
        result.passed / result.total,
        metadata={
            "passed": result.passed,
            "total": result.total,
        },
        feedback=result.summary,
    )
```

Sandboxing, resource limits, and execution policy belong to the application.
The reward function adapts the result into the common score contract.

### Tool-use rewards

```python
class ToolTraceInput(BaseModel):
    requested_tool: str
    called_tool: str | None
    arguments_valid: bool
    task_completed: bool


grader = RewardGrader(
    {
        "selected_tool": lambda item: float(
            item.called_tool == item.requested_tool
        ),
        "valid_arguments": lambda item: float(item.arguments_valid),
        "completed_task": lambda item: float(item.task_completed),
    },
    input_space=PydanticSpace(ToolTraceInput),
    weights={
        "selected_tool": 0.25,
        "valid_arguments": 0.25,
        "completed_task": 1.0,
    },
)
```

## Errors and failure behavior

The grader raises rather than silently changing a reward when:

- the input is outside `input_space`;
- no reward functions are configured;
- a component name is empty;
- a weight is invalid or names an unknown component;
- every configured weight is zero;
- a sync reward function returns an awaitable;
- a function or aggregator returns a value that cannot become a finite
  `Score`.

Exceptions raised by reward functions propagate to the caller. Catch and
convert failures inside a reward function only when the fallback behavior is a
deliberate part of the task's reward semantics.

## When to use another abstraction

Use a `RubricGrader` when the policy is primarily a portable human-defined
rubric interpreted by a judge.

Use a `CompositeGrader` when independently reusable graders should remain
visible as nested results. Do not flatten complete rubric and reward graders
into one large reward function mapping merely to combine their scalars.

## Related documentation

- [Grader concepts](../concepts/graders.md)
- [Composite graders](composite-graders.md)
- [Scores and aggregation](scores-and-aggregation.md)
- [Representative examples](examples.md)
