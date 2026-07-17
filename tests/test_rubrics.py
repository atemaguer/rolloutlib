from __future__ import annotations

import json
from typing import Any, cast

import pytest
from pydantic import ValidationError

from rolloutlib.graders import Criterion, Level, Rubric


def test_rubric_is_versioned_and_json_serializable() -> None:
    rubric = Rubric(
        id="answer-rubric",
        version="2",
        title="Answer quality",
        criteria=(
            Criterion(
                id="correctness",
                title="Correctness",
                description="The response is correct.",
                references=("Reference answer",),
                levels=(
                    Level(
                        id="correct",
                        label="Correct",
                        description="The response is fully correct.",
                        score=1.0,
                    ),
                    Level(
                        id="incorrect",
                        label="Incorrect",
                        description="The response is incorrect.",
                        score=0.0,
                    ),
                ),
            ),
        ),
        instructions="Use the reference answer.",
        metadata={"owner": "tests"},
    )

    encoded = rubric.model_dump_json()
    restored = Rubric.model_validate_json(encoded)

    assert restored == rubric
    assert restored.schema_version == "1"
    assert restored.criterion("correctness").level("correct").score == 1.0
    assert len(restored.fingerprint) == 64
    assert json.loads(encoded)["metadata"] == {"owner": "tests"}


def test_rubric_fingerprint_excludes_identity_but_tracks_content() -> None:
    criterion = Criterion(id="correct", description="The answer is correct.")
    first = Rubric(id="first", version="1", criteria=(criterion,))
    second = Rubric(id="second", version="2", criteria=(criterion,))
    changed = Rubric(
        id="first",
        version="1",
        criteria=(Criterion(id="correct", description="The answer is exact."),),
    )

    assert first.fingerprint == second.fingerprint
    assert first.fingerprint != changed.fingerprint


def test_rubric_produces_a_portable_json_schema() -> None:
    schema = Rubric.model_json_schema()

    assert schema["properties"]["schema_version"]["const"] == "1"
    assert "Criterion" in schema["$defs"]
    assert "Level" in schema["$defs"]


def test_rubric_rejects_empty_or_duplicate_criteria() -> None:
    with pytest.raises(ValidationError, match="at least 1"):
        Rubric(criteria=())

    with pytest.raises(ValidationError, match="unique"):
        Rubric(
            criteria=(
                Criterion(id="same", description="First."),
                Criterion(id="same", description="Second."),
            )
        )


def test_criterion_rejects_invalid_values_and_duplicate_levels() -> None:
    with pytest.raises(ValidationError):
        Criterion(id="", description="Valid.")
    with pytest.raises(ValidationError):
        Criterion(id="valid", description="Valid.", weight=0)
    with pytest.raises(ValidationError):
        Level(id="invalid", description="Invalid.", score=1.1)
    with pytest.raises(ValidationError, match="level ids"):
        Criterion(
            id="valid",
            description="Valid.",
            levels=(
                Level(id="same", description="Yes.", score=1.0),
                Level(id="same", description="No.", score=0.0),
            ),
        )
    with pytest.raises(ValidationError, match="level scores"):
        Criterion(
            id="valid",
            description="Valid.",
            levels=(
                Level(id="one", description="One.", score=1.0),
                Level(id="also-one", description="Also one.", score=1.0),
            ),
        )


def test_rubric_metadata_must_be_json_compatible() -> None:
    with pytest.raises(ValidationError):
        Rubric(
            criteria=(Criterion(id="valid", description="Valid."),),
            metadata=cast(Any, {"invalid": object()}),
        )
