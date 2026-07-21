# Graders

A grader turns an application-defined input into a structured `Score`.
Rolloutlib standardizes this boundary so environments, evaluation loops, and
training systems can consume grading signals without knowing how those signals
were produced.

The input can be anything an application needs to assess:

- one model response;
- a prompt, response, and reference answer;
- an environment action;
- a complete trajectory;
- a tool call and its result;
- a pair or group of candidate responses.

The abstraction deliberately does not prescribe a universal grading record or
model provider.

## The common contract

For immediate collaborators, grading is direct:

```python
score = grader.grade(input)
```

When a judge, reward function, or child grader is asynchronous, the same
method returns an awaitable:

```python
score = await grader.grade(input)
```

The resolved result is always a `Score`. The rubric, reward functions, model
client, and aggregation policy are configuration of a particular grader; they
are not additional arguments to `grade`.

This narrow contract is the point of the abstraction. Code that consumes a
grader only needs to know:

1. what value belongs to its `input_space`;
2. whether the particular operation returned an awaitable;
3. that the resolved result is a `Score`.

## Input spaces

Every grader declares a Gymnasium-compatible `input_space`. This plays the same
role as an environment's action and observation spaces: it documents the
accepted values and validates them at the public boundary.

For a response-only grader, a text space is enough:

```python
from rolloutlib import spaces

answer_space = spaces.text.text(min_length=1, max_length=8_000)
```

For a richer input, define an application model and wrap it in
`PydanticSpace`:

```python
from pydantic import BaseModel

from rolloutlib.spaces import PydanticSpace


class GradingInput(BaseModel):
    prompt: str
    response: str
    reference_answer: str


grading_input_space = PydanticSpace(GradingInput)
```

The space describes the complete value supplied to `grade`. It is not limited
to the generated response. This lets different domains define appropriate
records while preserving one grader contract.

Input validation occurs before any reward function, judge, or child grader is
invoked. An invalid value raises `ValueError`.

## The three grader families

Rolloutlib provides three concrete families:

| Family | Component scores come from | Default aggregation | Use it for |
| --- | --- | --- | --- |
| `RubricGrader` | one judge applying a bound `Rubric` | weighted mean | qualitative, human-defined, or LLM-mediated judgment |
| `RewardGrader` | named reward functions | weighted sum | exact checks, tests, validators, and heuristics |
| `CompositeGrader` | complete child graders | weighted mean | hybrid rewards and reusable grading pipelines |

Each family accepts synchronous or asynchronous collaborators.

The families are complementary rather than mutually exclusive. A common
post-training reward uses:

- a rubric grader for answer quality;
- a reward grader for deterministic correctness and formatting checks;
- a composite grader to combine the two without losing either result.

## Rubric graders

A rubric is portable data describing what should be judged. A rubric grader
binds that data to a user-owned judge:

```python
from rolloutlib.graders import RubricGrader

grader = RubricGrader(
    rubric,
    judge,
    input_space=grading_input_space,
)
```

The judge receives `(input, rubric)` and returns a mapping from every criterion
ID to a scalar or `Score`. It is called once per grading operation, so it can
make one holistic model request or coordinate several specialized checks.

Rubric graders are particularly useful for LLM judges, but they do not own the
model interaction. The application remains responsible for provider clients,
prompts, structured outputs, retries, caching, and tracing.

See [Rubrics and rubric graders](../graders/rubrics.md) for the complete schema,
JSON interchange, judge contract, metadata, and model-mediated examples.

## Reward graders

A reward grader evaluates separately named programmatic functions:

```python
from rolloutlib.graders import RewardGrader

grader = RewardGrader(
    {
        "exact_match": exact_match,
        "valid_format": valid_format,
    },
    input_space=grading_input_space,
    weights={"exact_match": 1.0, "valid_format": 0.1},
)
```

Each reward function receives the input and returns a scalar or `Score`. The
named outputs become visible components of the result.

See [Reward graders](../graders/reward-graders.md) for function contracts,
weights, custom aggregation, async execution, and examples.

## Composite graders

A composite grader evaluates complete child graders and retains their score
trees:

```python
from rolloutlib.graders import CompositeGrader

grader = CompositeGrader(
    {
        "quality": rubric_grader,
        "verification": reward_grader,
    },
    input_space=grading_input_space,
    weights={"quality": 0.8, "verification": 0.2},
)
```

Every child receives the same input. If the children need different context,
define a shared input record containing all required fields.

See [Composite graders](../graders/composite-graders.md) for nesting,
sync/async composition, environment wrappers, and hybrid reward design.

## Scores are data, not only rewards

`Score.value` is the scalar reward. `Score.components` contains named child
scores, allowing a result to preserve how that scalar was produced:

```python
from rolloutlib.graders import Score

score = Score(
    0.9,
    {
        "quality": Score(
            0.8,
            {
                "correctness": Score(0.75, feedback="One reasoning gap."),
                "format": Score(1.0),
            },
        ),
        "verification": Score(
            1.0,
            {"exact_match": Score(1.0)},
        ),
    },
)
```

The recursive structure supports both training and diagnosis: optimizers can
consume the scalar while evaluation and observability systems inspect
components, feedback, and metadata.

See [Scores and aggregation](../graders/scores-and-aggregation.md) for
serialization, environment `info`, built-in aggregators, formulas, and custom
policies.

## Synchronous and asynchronous grading

Use the same family whether work is local or asynchronous. `RubricGrader`
accepts a sync or async judge, `RewardGrader` accepts sync or async reward
functions, and `CompositeGrader` accepts children with either calling style.
When any collaborator is asynchronous, `grade` returns an awaitable; otherwise
it returns `Score` directly. Independent awaitable reward functions and child
graders are scheduled concurrently.

## Public value and callable types

The public type aliases make extension points explicit:

| Type | Meaning |
| --- | --- |
| `ScoreValue` | `float \| Score` |
| `RewardFunction[InputT]` | `InputT -> ScoreValue` or an awaitable score |
| `RubricJudge[InputT]` | `(InputT, Rubric) -> criterion mapping` or an awaitable mapping |
| `ScoreAggregator` | named component scores to a scalar |
| `RubricAggregator` | rubric plus criterion scores to a scalar |

These aliases are optional conveniences for annotations; ordinary compatible
callables work without explicitly importing them.

## Extending the contract

Applications can define a specialized grader when the standard families do not
fit. The public `grade` method owns input validation; subclasses implement
`_grade`:

```python
from rolloutlib import Grader, Score, spaces


class ExactMatchGrader(Grader[str]):
    input_space = spaces.text.text(min_length=1)

    def _grade(self, input: str) -> Score:
        return Score(float(input.strip() == "42"))
```

Use a custom subclass for a genuinely new grading protocol. Prefer
`RewardGrader` when the behavior is simply one or more named functions, because
the standard implementation preserves component names, weights, and metadata.

## Ownership boundaries

Rolloutlib owns:

- input validation through `input_space`;
- the `grade(input) -> Score` contract;
- portable rubric and criterion schemas;
- component normalization and aggregation;
- structured score serialization;
- sync and async composition.

Application code owns:

- the shape of domain-specific grader inputs;
- model providers and inference clients;
- prompts and structured-output parsing;
- retries, rate limiting, caching, and tracing;
- calibration data and human evaluation;
- the final choice of reward scaling and aggregation.

This separation keeps grader implementations portable across inference and
training backends.

## Next steps

- [Rubrics and rubric graders](../graders/rubrics.md)
- [Reward graders](../graders/reward-graders.md)
- [Composite graders](../graders/composite-graders.md)
- [Scores and aggregation](../graders/scores-and-aggregation.md)
- [Representative examples](../graders/examples.md)
- [Grader API reference](../api.md#graders)
