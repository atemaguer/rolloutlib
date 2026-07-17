# Scores and aggregation

`Score` is the common output of every grader. It carries a scalar reward and a
recursive record of how that reward was produced.

## Score fields

| Field | Type | Meaning |
| --- | --- | --- |
| `value` | finite `float` | Scalar result consumed as reward |
| `components` | `Mapping[str, Score]` | Named child results |
| `metadata` | `Mapping[str, Any]` | Provenance and application data |
| `feedback` | `str \| None` | Human-readable explanation |

Only `value` is required:

```python
from rolloutlib.graders import Score

score = Score(1.0)
```

Floats supplied as components are normalized automatically:

```python
score = Score(
    0.75,
    {
        "correctness": 1.0,
        "style": 0.5,
    },
)

assert score.components["correctness"] == Score(1.0)
```

Values must be finite. `NaN`, positive infinity, and negative infinity are
rejected.

## Feedback and metadata

Use `feedback` for a concise explanation of the judgment:

```python
score = Score(
    0.5,
    feedback="The conclusion is correct, but the proof skips a required step.",
)
```

Use `metadata` for structured provenance or measurements:

```python
score = Score(
    0.8,
    metadata={
        "judge_model": "application-model-id",
        "latency_ms": 420,
        "prompt_version": "3",
    },
)
```

Score metadata is application-defined and may contain arbitrary Python values.
If a score will be serialized to JSON, keep those values JSON-compatible.

Feedback must be non-empty after whitespace is removed.

## Recursive components

Components are scores, so results naturally form a tree:

```python
score = Score(
    0.9,
    {
        "quality": Score(
            0.8,
            {
                "correctness": Score(
                    0.75,
                    feedback="One reasoning step is unsupported.",
                ),
                "format": Score(1.0),
            },
        ),
        "verification": Score(
            1.0,
            {
                "exact_match": Score(1.0),
                "valid_json": Score(1.0),
            },
        ),
    },
)
```

Use components for named evidence that contributes to or explains a result.
Use metadata for contextual facts that are not themselves scored.

`component_values` provides the immediate scalar mapping:

```python
assert score.component_values == {
    "quality": 0.8,
    "verification": 1.0,
}
```

It does not flatten nested descendants.

## Normalizing scalar results

`Score.from_value` accepts either a float or an existing score:

```python
assert Score.from_value(0.5) == Score(0.5)

existing = Score(1.0, feedback="Passed.")
assert Score.from_value(existing) is existing
```

Graders use this operation to normalize user functions without discarding rich
results.

## Serialization

`to_dict()` produces a recursive dictionary:

```python
payload = score.to_dict()
restored = Score.from_dict(payload)

assert restored == score
```

The serialized shape is:

```python
{
    "value": 0.9,
    "components": {
        "quality": {
            "value": 0.8,
            "components": {},
            "metadata": {},
        },
    },
    "metadata": {},
}
```

`feedback` appears only when it is present.

## Environment `info`

`Score.as_info()` returns the conventional Gymnasium information payload:

```python
info = score.as_info()
assert info == {"score": score.to_dict()}
```

Read it back with `Score.from_info`:

```python
restored = Score.from_info(info)
assert restored == score
```

An absent score returns `None`, or a supplied default:

```python
assert Score.from_info({}) is None
assert Score.from_info({}, default=Score(0.0)) == Score(0.0)
```

Environment wrappers and single-turn environments use this convention so a
trainer can consume the scalar reward while logs and evaluation code retain
the structured result.

## Aggregation by grader family

Aggregation converts completed component scores into the parent scalar.

| Grader | Default | Rationale |
| --- | --- | --- |
| `RubricGrader` | weighted mean | criterion weights express relative importance |
| `RewardGrader` | weighted sum | programmatic signals commonly shape reward additively |
| `CompositeGrader` | weighted mean | child grader scales should remain stable under composition |

Aggregation never removes components. Changing the aggregate changes
`Score.value`, not the grading record.

## Rubric weighted mean

The default rubric aggregate is:

```text
value = Σ criterion.weight × criterion_score
        ──────────────────────────────────────
                  Σ criterion.weight
```

```python
from rolloutlib.graders import weighted_mean

value = weighted_mean(rubric, criterion_scores)
```

With correctness weighted `4.0`, format weighted `1.0`, and values `0.75` and
`1.0`, the result is:

```text
(4 × 0.75 + 1 × 1.0) / 5 = 0.8
```

## Built-in rubric aggregators

### `weighted_sum`

```python
from rolloutlib.graders import weighted_sum
```

Returns the unnormalized sum of criterion weight times criterion value. This is
useful when rubric criteria are intended as additive reward signals.

### `weighted_mean`

```python
from rolloutlib.graders import weighted_mean
```

Returns the normalized weighted average and is the rubric grader default.

### `all_pass`

```python
from rolloutlib.graders import all_pass
```

Returns `1.0` only when every criterion value is at least `1.0`; otherwise it
returns `0.0`. It is useful for strict acceptance rubrics.

### `asymmetric_mean`

```python
from rolloutlib.graders import asymmetric_mean
```

Supports criteria categorized as `required`, `bonus`, or `penalty`:

```text
required mean
+ bonus_weight × bonus mean
- penalty_weight × penalty failure mean
```

Uncategorized criteria are treated as `required`. Penalty criteria should be
phrased as desired behavior: their degree of failure is subtracted.

```python
from functools import partial

aggregate = partial(
    asymmetric_mean,
    bonus_weight=0.25,
    penalty_weight=2.0,
)
```

Unknown categories raise `ValueError`.

## Custom rubric aggregation

A rubric aggregator receives both the rubric and completed criterion scores:

```python
def correctness_gate(rubric, scores) -> float:
    if scores["correctness"].value < 1.0:
        return 0.0
    return weighted_mean(rubric, scores)


grader = RubricGrader(
    rubric,
    judge,
    input_space=grading_input_space,
    aggregate=correctness_gate,
)
```

Receiving the rubric lets the function inspect weights, categories, and
metadata.

## Custom reward and composite aggregation

Reward and composite aggregators receive only the completed named scores:

```python
def geometric_gate(scores) -> float:
    if any(score.value <= 0.0 for score in scores.values()):
        return 0.0
    product = 1.0
    for score in scores.values():
        product *= score.value
    return product ** (1 / len(scores))
```

The function returns a scalar. `Score` validates that the result is finite.

When a custom aggregate is configured, default weights are not automatically
applied. Capture weights in a closure or read component names explicitly when
the custom policy needs them.

## Choosing score ranges

Rolloutlib requires finite values but does not globally constrain scores to
`[0, 1]`.

- `Level.score` is constrained to `[0, 1]`.
- Rubric judges may technically return other finite values.
- Reward functions commonly return bonuses or penalties outside that range.
- Composite graders combine whatever scale their children expose.

Document expected ranges and normalize before composition when downstream
training code assumes a particular scale.

## Metadata conventions

Rolloutlib automatically attaches rubric identity metadata to rubric grader
results:

- `rubric_fingerprint`;
- `rubric_id`, when configured;
- `rubric_version`, when configured.

Applications may add conventions such as:

- `grader_version`;
- `judge_model`;
- `prompt_version`;
- `reference_version`;
- `calibration_set`;
- `latency_ms`.

Prefer stable machine-readable fields. Put longer natural-language reasoning in
`feedback`.

## Related documentation

- [Grader concepts](../concepts/graders.md)
- [Rubrics and rubric graders](rubrics.md)
- [Reward graders](reward-graders.md)
- [Composite graders](composite-graders.md)
- [Representative examples](examples.md)
