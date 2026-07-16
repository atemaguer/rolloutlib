"""Small, environment-backed benchmark definitions."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any

from ..core import Benchmark
from .math import (
    AIMEEnv,
    AIME_PROMPT_SUFFIX,
    GSM8KEnv,
    MathEnv,
    MathExample,
    aime,
    extract_aime_answer,
    extract_gsm8k_answer,
    grade_aime,
    grade_gsm8k,
    gsm8k,
    make_example_id,
)


BenchmarkFactory = Callable[..., Benchmark[MathExample]]
REGISTRY: Mapping[str, BenchmarkFactory] = {
    "aime": aime,
    "gsm8k": gsm8k,
}


def make(
    name: str,
    examples: Iterable[MathExample | Mapping[str, Any]] | None = None,
    **kwargs: Any,
) -> Benchmark[MathExample]:
    """Construct a registered benchmark by name."""

    try:
        factory = REGISTRY[name]
    except KeyError as exc:
        raise ValueError(
            f"unknown benchmark {name!r}; choose from {sorted(REGISTRY)}"
        ) from exc
    if examples is None:
        return factory(**kwargs)
    return factory(examples, **kwargs)


__all__ = [
    "AIMEEnv",
    "AIME_PROMPT_SUFFIX",
    "BenchmarkFactory",
    "GSM8KEnv",
    "MathEnv",
    "MathExample",
    "REGISTRY",
    "aime",
    "extract_aime_answer",
    "extract_gsm8k_answer",
    "grade_aime",
    "grade_gsm8k",
    "gsm8k",
    "make",
    "make_example_id",
]
