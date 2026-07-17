# Rubrics and rubric graders

A `Rubric` is a portable description of human-defined judgment criteria. A
`RubricGrader` binds that description to a judge that knows how to assess an
input.

This separation lets the same rubric be:

- stored with a dataset;
- serialized and shared independently of code;
- applied by different models or human reviewers;
- aggregated differently for evaluation and training;
- versioned and audited over time.

## The rubric schema

Rubrics are strict, frozen Pydantic models. Unknown fields are rejected and the
models publish JSON Schema through `model_json_schema()`.

The schema has three levels:

```text
Rubric
└── Criterion
    └── Level
```

The hierarchy is descriptive. Scored criteria remain flat and independently
addressable by criterion ID.

## Levels

A `Level` describes one named performance band for a criterion.

| Field | Type | Required | Meaning |
| --- | --- | --- | --- |
| `id` | `str` | yes | Stable identifier within the criterion |
| `description` | `str` | yes | Observable meaning of this level |
| `score` | `float` | yes | Normalized value from `0.0` through `1.0` |
| `label` | `str \| None` | no | Human-readable display name |

Level IDs and scores must be unique within a criterion. IDs, descriptions, and
labels cannot be empty after whitespace is removed.

```python
from rolloutlib import Level

complete = Level(
    id="complete",
    label="Complete",
    description="The conclusion and supporting reasoning are correct.",
    score=1.0,
)
```

Levels are guidance for a judge; the result still contains a numeric `Score`.
They work well for discrete classroom-style bands. Omit them when a criterion
is more naturally scored continuously.

## Criteria

A `Criterion` describes one independently assessable requirement.

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `id` | `str` | required | Stable component name in judge results |
| `description` | `str` | required | The requirement to assess |
| `weight` | positive `float` | `1.0` | Relative importance during rubric aggregation |
| `title` | `str \| None` | `None` | Human-readable display name |
| `levels` | `tuple[Level, ...]` | empty | Optional performance bands |
| `category` | `str \| None` | `None` | Optional grouping or aggregation vocabulary |
| `references` | `tuple[str, ...]` | empty | Reference material or identifiers |
| `metadata` | JSON object | empty | Application-specific portable data |

```python
from rolloutlib import Criterion, Level

correctness = Criterion(
    id="correctness",
    title="Correctness",
    description="The response reaches the correct conclusion and supports it.",
    weight=4.0,
    levels=(
        Level(
            id="complete",
            description="The conclusion and reasoning are correct.",
            score=1.0,
        ),
        Level(
            id="partial",
            description="The conclusion is correct with a material reasoning gap.",
            score=0.5,
        ),
        Level(
            id="incorrect",
            description="The conclusion is incorrect.",
            score=0.0,
        ),
    ),
)
```

Criterion IDs must be unique within a rubric. Weights must be finite and
positive.

Criteria and rubrics provide identifier lookup helpers:

```python
criterion = rubric.criterion("correctness")
level = criterion.level("complete")
```

An unknown identifier raises `KeyError`.

### Why criteria remain flat

Rubric documents often display nested sections, but recursive scoring introduces
unclear semantics: a parent may be a container, an independently scored item,
or both. Rolloutlib keeps scored criteria flat so every result has one stable
address.

Use `category` and `metadata` to preserve presentation or reporting groups:

```python
Criterion(
    id="citation_quality",
    description="Citations support the claims they are attached to.",
    category="evidence",
    metadata={"section": "Research quality", "order": 2},
)
```

## Rubrics

A `Rubric` packages criteria and shared instructions.

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `schema_version` | `"1"` | `"1"` | Version of Rolloutlib's interchange schema |
| `id` | `str \| None` | `None` | Stable published identity |
| `version` | `str \| None` | `None` | Application-defined rubric version |
| `title` | `str \| None` | `None` | Human-readable title |
| `description` | `str \| None` | `None` | Purpose and intended use |
| `instructions` | `str \| None` | `None` | Guidance shared by all criteria |
| `criteria` | non-empty tuple | required | Independently scored criteria |
| `metadata` | JSON object | empty | Application-specific portable data |

```python
from rolloutlib import Criterion, Level, Rubric

rubric = Rubric(
    id="answer-quality",
    version="1.0",
    title="Answer quality",
    description="Evaluates factual and presentational quality.",
    instructions=(
        "Assess only the submitted response. Do not reward information that is "
        "present only in the reference material."
    ),
    criteria=(
        Criterion(
            id="correctness",
            description="The response reaches the correct conclusion.",
            weight=4.0,
            levels=(
                Level(
                    id="correct",
                    description="The conclusion is correct.",
                    score=1.0,
                ),
                Level(
                    id="incorrect",
                    description="The conclusion is incorrect.",
                    score=0.0,
                ),
            ),
        ),
        Criterion(
            id="reasoning",
            description="The reasoning is valid and supports the conclusion.",
            weight=2.0,
        ),
        Criterion(
            id="format",
            description="The requested format is followed.",
            weight=1.0,
        ),
    ),
    metadata={"domain": "mathematics"},
)
```

## JSON interchange and identity

Pydantic supplies JSON serialization, validation, and schema generation:

```python
encoded = rubric.model_dump_json(indent=2)
restored = Rubric.model_validate_json(encoded)
json_schema = Rubric.model_json_schema()

assert restored == rubric
```

`Rubric.fingerprint` is a stable SHA-256 digest of rubric content. The published
`id` and `version` fields are excluded, so two differently named rubric records
with identical evaluative content have the same fingerprint.

```python
fingerprint = rubric.fingerprint
```

Use the fields for different purposes:

- `schema_version` tells a reader how to interpret the JSON format;
- `id` identifies the rubric in an application or registry;
- `version` identifies an application-managed release;
- `fingerprint` identifies the actual evaluative content.

All rubric metadata must be JSON-compatible so the complete rubric remains
portable.

## The rubric judge contract

A synchronous rubric judge has the shape:

```python
def judge(input, rubric) -> Mapping[str, float | Score]:
    ...
```

An asynchronous judge may return that mapping directly or await it:

```python
async def judge(input, rubric) -> Mapping[str, float | Score]:
    ...
```

The mapping must contain exactly one entry for every criterion ID. Missing or
unknown IDs raise `ValueError`; a non-mapping result raises `TypeError`.

Returning a scalar is convenient:

```python
return {
    "correctness": 1.0,
    "reasoning": 0.75,
    "format": 1.0,
}
```

Returning `Score` preserves feedback and criterion-specific metadata:

```python
from rolloutlib.graders import Score

return {
    "correctness": Score(
        1.0,
        feedback="The final result matches the reference answer.",
    ),
    "reasoning": Score(
        0.75,
        metadata={"selected_level": "mostly_correct"},
        feedback="One algebraic step is asserted without explanation.",
    ),
    "format": Score(1.0),
}
```

The judge is invoked once per input. This is intentional: an LLM judge can
assess all criteria in one coherent request, while a more complex judge can
coordinate several calls internally.

## Constructing a rubric grader

```python
from rolloutlib.graders import RubricGrader

grader = RubricGrader(
    rubric,
    judge,
    input_space=grading_input_space,
)

score = grader.grade(item)
```

Constructor options:

| Argument | Meaning |
| --- | --- |
| `rubric` | The immutable rubric applied by this grader |
| `judge` | User-owned callable returning criterion scores |
| `input_space` | Space describing accepted grader inputs |
| `aggregate` | Optional rubric aggregation function |
| `metadata` | Metadata attached to every top-level result |

The default aggregation is `weighted_mean`. A successful result automatically
includes:

- `rubric_fingerprint`;
- `rubric_id`, when present;
- `rubric_version`, when present.

The criterion results are stored under `score.components`.

## LLM-mediated rubric grading

Rubric graders are model-provider-neutral. The application renders the request,
calls the provider, validates its structured response, and returns Rolloutlib
scores.

```python
from pydantic import BaseModel, Field

from rolloutlib.graders import AsyncRubricGrader, Rubric, Score


class CriterionResult(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    feedback: str


class JudgeResponse(BaseModel):
    criteria: dict[str, CriterionResult]


async def judge(input: GradingInput, rubric: Rubric):
    prompt = render_rubric_prompt(input, rubric)
    raw_response = await model_client.generate(prompt)
    response = JudgeResponse.model_validate_json(raw_response)
    return {
        criterion_id: Score(
            result.score,
            feedback=result.feedback,
            metadata={"judge_model": model_client.model_name},
        )
        for criterion_id, result in response.criteria.items()
    }


grader = AsyncRubricGrader(
    rubric,
    judge,
    input_space=grading_input_space,
    metadata={"grader": "answer-quality-judge"},
)
```

Rolloutlib intentionally leaves the following on the application side:

- provider SDK and model selection;
- prompt rendering;
- structured-output configuration and parsing;
- sampling settings;
- retries and rate limits;
- caching and tracing;
- judge calibration and monitoring.

This boundary permits hosted APIs, local inference, multimodal judges, human
review, and multi-model ensembles to implement the same judge contract.

## One rubric per grader

The rubric is bound at construction, making the grader's policy explicit and
preventing call sites from accidentally switching policies.

If each dataset item has its own rubric, create a grader when constructing that
item's environment:

```python
def make_grader(item: DatasetItem) -> AsyncRubricGrader[GradingInput]:
    return AsyncRubricGrader(
        item.rubric,
        judge,
        input_space=grading_input_space,
    )
```

This mirrors environment construction: dataset items configure fresh runtime
objects rather than passing configuration through every operation.

## Designing reliable rubrics

Good criteria are:

- independently assessable;
- specific about observable evidence;
- non-overlapping where possible;
- stable enough to version and compare over time;
- explicit about what should not affect the judgment.

Prefer criterion IDs that are stable machine identifiers, such as
`factual_correctness`, and titles that are suitable for display, such as
`Factual correctness`.

Levels should describe behavior, not only adjectives. “The answer includes all
required evidence” is more reproducible than “excellent.”

Before using an LLM-mediated score as a training reward:

- compare it with human judgments;
- test adversarial and reward-hacking examples;
- measure agreement separately by criterion;
- inspect sensitivity to prompt and model changes;
- retain rubric fingerprints and judge metadata with results.

## Related documentation

- [Grader concepts](../concepts/graders.md)
- [Scores and aggregation](scores-and-aggregation.md)
- [Representative examples](examples.md)
- [API reference](../api.md#graders)
