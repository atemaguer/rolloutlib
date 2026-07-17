# Graders and rubrics

A grader turns an application-defined input into a `Score`. The input may be a
response, action, trajectory, tool trace, comparison, or richer object
containing references and task state.

Grading has five concepts:

- `Grader` defines the synchronous `grade` operation.
- `AsyncGrader` provides the same value-level contract asynchronously.
- `input_space` describes the values a grader accepts.
- `Rubric` and `Criterion` describe what should be assessed.
- `Score` records the scalar result and its component results.

## The grader contract

Synchronous graders implement:

```python
score = grader.grade(input, rubric=rubric)
```

Async graders implement the same operation as an awaitable:

```python
score = await grader.grade(input, rubric=rubric)
```

Both always return `Score`. Rubrics are optional because deterministic
verifiers and reward functions may fully define their own grading behavior.
A grader that requires a rubric should reject a missing rubric explicitly.

Every grader declares a Gymnasium-compatible `input_space`. The public
`grade()` operation checks the input against this space before running any
deterministic check or model call. Invalid values raise `ValueError`.

```python
from rolloutlib import spaces
from rolloutlib.graders import CallableGrader

grader = CallableGrader(
    exact_match,
    input_space=spaces.text.text(min_length=1, max_length=1_000),
)

assert "42" in grader.input_space
score = grader.grade("42")
```

The space describes the whole input to the grader, not necessarily only the
candidate response. A task that needs context can define a `TypedDict`,
Pydantic model, or other type and use `PydanticSpace`:

```python
from typing import TypedDict

from rolloutlib.spaces import PydanticSpace


class GradingInput(TypedDict):
    prompt: str
    response: str
    reference_answer: str


grading_input_space = PydanticSpace(GradingInput)
```

This makes string graders, trajectory graders, preference graders, and richer
task-specific graders interoperable through the same contract without imposing
one universal input record. When a grader directly evaluates an environment
action or observation, it can reuse that environment space.

Custom graders normally declare the space alongside their grading behavior:

```python
from rolloutlib import Grader, Score, spaces


class ExactMatchGrader(Grader[str]):
    input_space = spaces.text.text(min_length=1)

    def _grade(self, input, *, rubric=None):
        return Score(float(input == "42"))
```

`grade()` remains the single public operation: it owns input validation and
then invokes the custom grading behavior.

Rubrics are passed at grading time because they may vary by dataset item. When
one rubric is fixed for many calls, bind it:

```python
bound = grader.bind(rubric)
score = bound.grade(input)
```

`CallableGrader` and `AsyncCallableGrader` adapt ordinary callables to the
standard contracts. The callable may contain deterministic logic, an LLM call,
or any other grading implementation.

## Rubrics as portable data

Rubrics are strict, frozen Pydantic models. They preserve criterion order,
round-trip through JSON, publish a JSON Schema, and provide a stable content
fingerprint.

```python
from rolloutlib import Criterion, Level, Rubric

rubric = Rubric(
    id="answer-quality",
    version="1",
    title="Answer quality",
    criteria=(
        Criterion(
            id="correctness",
            description="The answer reaches a correct conclusion.",
            weight=4.0,
            levels=(
                Level(
                    id="complete",
                    description="The conclusion and reasoning are correct.",
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
    ),
)

encoded = rubric.model_dump_json()
restored = Rubric.model_validate_json(encoded)
schema = Rubric.model_json_schema()
```

`schema_version` versions the interchange format. `id` and `version` identify a
particular published rubric but are excluded from its content fingerprint.
Metadata must contain JSON-compatible values.

## Criteria and levels

A criterion is one independently assessable requirement. Criteria intentionally
remain flat: `category` and metadata can organize them without making scoring
semantics recursive.

Levels are optional performance bands within a criterion. Their scores range
from zero to one. Criterion weight expresses relative importance separately
from the selected performance level. Criteria without levels can be scored
continuously.

Criterion-level results conventionally use criterion IDs as component names:

```python
Score(
    value=0.8,
    components={
        "correctness": Score(0.75),
        "format": Score(1.0),
    },
)
```

## Applying a rubric

`RubricGrader` applies a scorer to every criterion and combines the component
scores. A default scorer can handle every criterion while overrides handle
deterministic or specialized checks:

```python
from rolloutlib.graders import RubricGrader

grader = RubricGrader(
    llm_criterion_scorer,
    input_space=grading_input_space,
    overrides={"correctness": exact_answer_scorer},
)

score = grader.grade(input, rubric=rubric)
```

The default aggregation is a weighted mean. `weighted_sum`, `all_pass`,
`asymmetric_mean`, and custom aggregation callables are available.
`AsyncRubricGrader` supports sync or async criterion scorers and evaluates
independent criteria concurrently.

Aggregation is an implementation concern rather than part of the portable
rubric schema. This permits explicit weighted scoring, holistic model grading,
vetoes, penalties, and optimizer-specific reward shaping to use the same
rubric.

## Model-mediated grading

An LLM-mediated grader is not a separate grading contract. It is ordinarily an
`AsyncCallableGrader` whose callable renders a request, invokes an
application-owned model client, and parses the result into a `Score`:

```python
from rolloutlib.graders import AsyncCallableGrader


async def grade_with_judge(input: GradingInput, rubric: Rubric | None) -> Score:
    request = render_judge_request(input, rubric)
    response = await call_judge_model(request)  # application-owned
    return parse_judge_response(response)


grader = AsyncCallableGrader(
    grade_with_judge,
    input_space=grading_input_space,
)
```

This keeps provider SDKs, model selection, prompts, sampling settings, retries,
caching, structured outputs, and tracing in the application that owns them.
Rolloutlib supplies the shared input validation, rubric handling, and score
contract. The same pattern works for a hosted model, local inference, a
multimodal judge, or a sequence of several model calls.

Model graders should be calibrated against human judgments and adversarial
examples before their scores are used as rewards. Detailed, independently
assessable criteria improve diagnosis, but they do not make an unreliable judge
safe automatically.
