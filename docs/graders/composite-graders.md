# Composite graders

A `CompositeGrader` combines complete child graders into one score while
preserving every child's internal result. It is the composition primitive for
hybrid rewards and reusable grading pipelines.

## Why composition is distinct from reward functions

A reward grader exposes individual functions as one flat component mapping. A
composite grader preserves abstraction boundaries:

```text
final score
├── quality
│   ├── correctness
│   ├── reasoning
│   └── style
└── verification
    ├── exact_match
    └── valid_format
```

Here `quality` may be a rubric grader and `verification` may be a reward
grader. Their complete scores remain independently inspectable and reusable.

## A synchronous composite

```python
from rolloutlib.graders import CompositeGrader

grader = CompositeGrader(
    {
        "quality": rubric_grader,
        "verification": reward_grader,
    },
    input_space=grading_input_space,
    weights={
        "quality": 0.8,
        "verification": 0.2,
    },
)

score = grader.grade(item)
```

`CompositeGrader` accepts synchronous child graders. For sync and async
children together, use `CompositeGrader`.

Constructor options:

| Argument | Meaning |
| --- | --- |
| `graders` | Mapping from stable component names to child graders |
| `input_space` | Space describing the input shared by all children |
| `weights` | Optional non-negative child weights |
| `aggregate` | Optional function producing the final scalar |
| `metadata` | Metadata attached to every top-level result |

## The shared input contract

Every child receives the same input:

```python
child_score = child.grade(input)
```

The composite's `input_space` validates that value first, and each child then
applies its own input-space validation.

When children need different context, define one shared record:

```python
from pydantic import BaseModel

from rolloutlib.spaces import PydanticSpace


class EvaluationInput(BaseModel):
    prompt: str
    response: str
    reference_answer: str
    required_format: str


evaluation_space = PydanticSpace(EvaluationInput)
```

The rubric judge may read the prompt and response, while deterministic
functions read the reference answer and required format.

Composition does not adapt or route different inputs to children. If input
transformation is essential, define a domain-specific grader or wrap the child
in an application-owned adapter that still implements the grader contract.

## Names, weights, and default aggregation

Child grader names must be non-empty and stable. At least one child is required.
The names become top-level component keys.

The default aggregate is a weighted mean:

```text
value = Σ weight[name] × child[name].value
        ─────────────────────────────────────
                    Σ weight[name]
```

Every unspecified weight defaults to `1.0`.

Weights must be finite and non-negative. At least one must be positive, and a
weight cannot reference an unknown child name. A zero-weight child is still
evaluated and retained for diagnostics.

The weighted mean keeps the composite scale stable when normalized child
scores are added or removed. If a child intentionally produces unbounded or
additive rewards, account for that scale explicitly.

## Nested scores

The child `Score` is stored intact:

```python
score = grader.grade(item)

quality = score.components["quality"]
correctness = quality.components["correctness"]

print(score.value)
print(quality.value)
print(correctness.feedback)
```

Composite-level metadata is separate from child metadata:

```python
grader = CompositeGrader(
    {
        "quality": rubric_grader,
        "verification": reward_grader,
    },
    input_space=evaluation_space,
    metadata={
        "reward_policy": "quality-plus-verification",
        "reward_policy_version": "2",
    },
)
```

This makes provenance available at each layer.

## Awaitable composition

`CompositeGrader` accepts both synchronous and asynchronous children:

```python
from rolloutlib.graders import CompositeGrader

grader = CompositeGrader(
    {
        "quality": async_rubric_grader,
        "verification": reward_grader,
        "safety": async_safety_grader,
    },
    input_space=evaluation_space,
    weights={
        "quality": 0.7,
        "verification": 0.2,
        "safety": 0.1,
    },
)

score = await grader.grade(item)
```

Children are scheduled concurrently whenever any child is awaitable. The same
`CompositeGrader` remains immediate when every child is synchronous.
Long-running synchronous work can still block the event loop, so applications
should provide an async implementation for blocking I/O or offload it.

## Custom aggregation

A custom composite aggregator receives a mapping from child name to completed
`Score` and returns the final scalar:

```python
def safety_gate(scores) -> float:
    if scores["safety"].value < 1.0:
        return 0.0
    return (
        0.8 * scores["quality"].value
        + 0.2 * scores["verification"].value
    )


grader = CompositeGrader(
    {
        "quality": async_rubric_grader,
        "verification": reward_grader,
        "safety": async_safety_grader,
    },
    input_space=evaluation_space,
    aggregate=safety_gate,
)
```

When `aggregate` is supplied, it completely determines the top-level value.
Configured weights are not automatically applied inside it.

Aggregation should be a pure, inexpensive operation over completed child
scores. Put model calls and external checks in child graders instead.

## Composing more than one level

Composite graders can contain other composite graders:

```python
task_success = CompositeGrader(
    {
        "correctness": correctness_grader,
        "tool_use": tool_use_grader,
    },
    input_space=evaluation_space,
)

overall = CompositeGrader(
    {
        "task_success": task_success,
        "quality": quality_grader,
    },
    input_space=evaluation_space,
)
```

Use nesting when the intermediate grouping has meaning for reporting,
ownership, reuse, or aggregation. Avoid unnecessary layers that merely rename
one child.

## Grading inside environments

When a grade determines the environment reward, perform grading inside
`step`. `GradingWrapper` composes an environment with a grader:

```python
from rolloutlib import wrappers

environment = wrappers.GradingWrapper(
    ExistingEnv(item),
    grader=grader,
    make_input=lambda env, action: EvaluationInput(
        prompt=item.prompt,
        response=action,
        reference_answer=item.reference_answer,
        required_format=item.required_format,
    ),
    input_space=evaluation_space,
)
```

By default, the wrapper:

1. lets the inner environment process the action;
2. grades only terminated or truncated steps;
3. replaces the inner scalar reward with `score.value`;
4. serializes the full score under `info["score"]`.

The same wrapper accepts either a sync or async grader:

```python
from rolloutlib import wrappers

environment = wrappers.GradingWrapper(
    ExistingEnv(item),
    grader=async_grader,
    make_input=lambda env, action: EvaluationInput(
        prompt=item.prompt,
        response=action,
        reference_answer=item.reference_answer,
        required_format=item.required_format,
    ),
    input_space=evaluation_space,
)
```

### Choosing which steps to grade

The `when` callable receives `(terminated, truncated)`:

```python
environment = wrappers.GradingWrapper(
    ExistingEnv(item),
    grader=grader,
    make_input=make_input,
    input_space=grader.input_space,
    when=lambda terminated, truncated: terminated,
)
```

The default grades either terminal condition. Supply a different predicate for
task-specific semantics.

### Combining with an existing reward

The default replaces the inner reward. `combine_reward` receives the existing
scalar and complete `Score`:

```python
environment = wrappers.GradingWrapper(
    ExistingEnv(item),
    grader=grader,
    make_input=make_input,
    input_space=grader.input_space,
    combine_reward=lambda environment_reward, score: (
        environment_reward + score.value
    ),
)
```

Keep reward scaling explicit. Combining values from different ranges without a
documented policy makes training behavior difficult to interpret.

## Direct use in single-turn environments

`SingleTurnEnv.evaluate` may return a `Score` directly or an awaitable score:

```python
class AnswerEnv(SingleTurnEnv[str, str]):
    def evaluate(self, action: str):
        item = EvaluationInput(
            prompt=self.prompt,
            response=action,
            reference_answer=self.reference_answer,
            required_format=self.required_format,
        )
        return grader.grade(item)
```

The environment uses `Score.value` as reward and stores the full score in
`info`.

## Errors and failure behavior

The composite raises when:

- its input is outside `input_space`;
- no child graders are configured;
- a child name is empty;
- weights are invalid or reference unknown children;
- every configured weight is zero;
- a child grader raises;
- the aggregate does not produce a finite scalar.

Failures propagate instead of being converted to zero. A zero reward is a task
outcome, while a failed grader is an infrastructure or implementation event;
silently conflating them corrupts training data.

## Related documentation

- [Grader concepts](../concepts/graders.md)
- [Rubrics and rubric graders](rubrics.md)
- [Reward graders](reward-graders.md)
- [Scores and aggregation](scores-and-aggregation.md)
- [Representative examples](examples.md)
