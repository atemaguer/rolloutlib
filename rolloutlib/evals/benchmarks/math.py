"""Single-turn math benchmark environments.

The benchmark factories accept in-memory examples so rolloutlib does not need
to own a dataset dependency. Passing no examples loads the conventional
Hugging Face dataset lazily and requires the optional ``benchmarks`` extra.
"""

from __future__ import annotations

import hashlib
import importlib
import re
from collections.abc import Iterable, Mapping
from decimal import Decimal, InvalidOperation
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, Field

from ...envs import SingleTurnEnv
from ...graders import Score
from ...spaces import messages, text
from ...types import Chat
from ..core import Benchmark


AIME_PROMPT_SUFFIX = (
    "\n\nThis is an AIME problem. The answer is an integer from 000 to 999. "
    "Show your work step by step, then put your final answer in \\boxed{}."
)


_NUMBER = re.compile(r"[-+]?(?:\d[\d,]*)(?:\.\d+)?")
_GSM8K_MARKER = re.compile(r"####\s*([^\n]+)", re.IGNORECASE)
_ANSWER_MARKER = re.compile(r"(?:answer is|answer:)\s*\$?([0-9,.-]+)", re.IGNORECASE)


def _clean_answer(value: str) -> str:
    value = value.strip().replace("$", "").replace(",", "")
    value = value.replace("\\,", "")
    return value.strip().rstrip(".!?;:")


def _canonical_answer(value: str) -> str:
    value = _clean_answer(value)
    try:
        number = Decimal(value)
    except (InvalidOperation, ValueError):
        return " ".join(value.casefold().split())
    if not number.is_finite():
        return " ".join(value.casefold().split())
    return format(number.normalize(), "f")


def _extract_boxed(value: str) -> str | None:
    """Extract the contents of the first ``\\boxed{...}`` expression."""

    marker = value.find(r"\boxed{")
    if marker < 0:
        return None
    start = marker + len(r"\boxed{")
    depth = 1
    index = start
    while index < len(value) and depth:
        if value[index] == "{":
            depth += 1
        elif value[index] == "}":
            depth -= 1
        index += 1
    return value[start : index - 1] if depth == 0 else None


def _extract_number(value: str) -> str:
    """Extract the first numeric token after removing simple LaTeX syntax."""

    cleaned = re.sub(r"\\text\{[^}]*\}", "", value)
    cleaned = re.sub(r"\\[a-zA-Z]+", "", cleaned)
    cleaned = (
        cleaned.replace("{", "")
        .replace("}", "")
        .replace("$", "")
        .replace(",", "")
        .replace(" ", "")
    )
    match = re.search(r"[-]?\d+\.?\d*", cleaned)
    return match.group(0) if match else cleaned.strip()


def _grade_answer(response: str, expected: str, extracted: str | None) -> Score:
    correct = float(
        extracted is not None
        and _canonical_answer(extracted) == _canonical_answer(expected)
    )
    return Score(
        value=correct,
        components={"correct": correct},
        metadata={"expected": _clean_answer(expected), "extracted": extracted},
    )


def extract_gsm8k_answer(response: str) -> str | None:
    """Extract a final numeric answer from a GSM8K-style response."""

    boxed = _extract_boxed(response)
    if boxed:
        return _clean_answer(_extract_number(boxed))
    marked = _GSM8K_MARKER.search(response)
    if marked:
        return _clean_answer(_extract_number(marked.group(1)))
    answer = _ANSWER_MARKER.search(response)
    if answer:
        return _clean_answer(answer.group(1))
    bold = re.findall(r"\*\*\$?([-]?\d+[\d,]*\.?\d*)", response)
    if bold:
        return _clean_answer(bold[-1])
    numbers = _NUMBER.findall(response)
    return _clean_answer(numbers[-1]) if numbers else None


def extract_aime_answer(response: str) -> str | None:
    """Extract the final 0--999 integer from an AIME-style response."""

    boxed = _extract_boxed(response)
    if boxed:
        return _clean_answer(_extract_number(boxed))
    return extract_gsm8k_answer(response)


def grade_gsm8k(response: str, expected: str) -> Score:
    """Grade a response against a GSM8K answer field."""

    expected_answer = extract_gsm8k_answer(expected) or expected
    return _grade_answer(response, expected_answer, extract_gsm8k_answer(response))


def grade_aime(response: str, expected: str) -> Score:
    """Grade a response against an AIME integer answer."""

    expected_answer = extract_aime_answer(expected) or expected
    return _grade_answer(response, expected_answer, extract_aime_answer(response))


class MathExample(BaseModel):
    """A problem and its reference answer."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    question: str
    answer: str
    example_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


def make_example_id(prefix: str, question: str) -> str:
    """Create a stable content-derived example identifier."""

    digest = hashlib.sha256(question.strip().encode("utf-8")).hexdigest()[:16]
    return f"{prefix}:{digest}"


def _coerce_example(
    value: MathExample | Mapping[str, Any],
    *,
    question_keys: tuple[str, ...],
    answer_keys: tuple[str, ...],
    prefix: str,
) -> MathExample:
    if isinstance(value, MathExample):
        if value.example_id is not None:
            return value
        return value.model_copy(
            update={"example_id": make_example_id(prefix, value.question)}
        )
    row = dict(value)
    question = next((row[key] for key in question_keys if key in row), None)
    answer = next((row[key] for key in answer_keys if key in row), None)
    if not isinstance(question, str):
        raise ValueError(
            f"benchmark rows must contain string fields from {question_keys!r} "
            f"and {answer_keys!r}"
        )
    if not isinstance(answer, str):
        if isinstance(answer, (int, float)):
            answer = str(answer)
        else:
            raise ValueError(
                f"benchmark answer must be a string or number, got {answer!r}"
            )
    example_id = row.get("example_id") or row.get("id")
    if example_id is not None:
        example_id = str(example_id)
    metadata = {
        key: item
        for key, item in row.items()
        if key not in {*question_keys, *answer_keys, "example_id", "id"}
    }
    return MathExample(
        question=question,
        answer=answer,
        example_id=example_id or make_example_id(prefix, question),
        metadata=metadata,
    )


def _load_huggingface(
    dataset: str,
    *,
    config: str | None,
    split: str,
) -> Iterable[Mapping[str, Any]]:
    try:
        load_dataset = importlib.import_module("datasets").load_dataset
    except ImportError as exc:
        raise ImportError(
            "loading built-in benchmarks requires the optional dependency; "
            "install with `pip install rolloutlib[benchmarks]`"
        ) from exc
    if config is None:
        return load_dataset(dataset, split=split)
    return load_dataset(dataset, config, split=split)


Grade = Callable[[str, str], Score]


class MathEnv(SingleTurnEnv[Chat, str]):
    """A single-turn chat environment for a math answer."""

    def __init__(
        self,
        example: MathExample,
        grade: Grade,
        *,
        system_prompt: str | None = None,
        prompt_suffix: str = "",
    ) -> None:
        super().__init__()
        self.example = example
        self._grade = grade
        self.system_prompt = system_prompt
        self.prompt_suffix = prompt_suffix
        self.action_space = text.text()
        self.observation_space = messages.chat(min_length=0)

    def initial_observation(
        self, *, options: dict[str, Any] | None = None
    ) -> tuple[Chat, dict[str, Any]]:
        del options
        messages: Chat = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append(
            {"role": "user", "content": self.example.question + self.prompt_suffix}
        )
        return messages, {"example_id": self.example.example_id}

    def evaluate(self, action: str) -> tuple[Score, dict[str, Any]]:
        score = self._grade(action, self.example.answer)
        return score, {"example_id": self.example.example_id}

    def terminal_observation(self, action: str) -> Chat:
        return [{"role": "assistant", "content": action}]


class GSM8KEnv(MathEnv):
    """Single-turn GSM8K environment."""

    def __init__(
        self,
        example: MathExample,
        *,
        system_prompt: str | None = None,
    ) -> None:
        super().__init__(example, grade_gsm8k, system_prompt=system_prompt)


class AIMEEnv(MathEnv):
    """Single-turn AIME environment."""

    def __init__(
        self,
        example: MathExample,
        *,
        system_prompt: str | None = None,
        prompt_suffix: str = AIME_PROMPT_SUFFIX,
    ) -> None:
        super().__init__(
            example,
            grade_aime,
            system_prompt=system_prompt,
            prompt_suffix=prompt_suffix,
        )


def gsm8k(
    examples: Iterable[MathExample | Mapping[str, Any]] | None = None,
    *,
    split: str = "test",
    dataset: str = "openai/gsm8k",
    config: str = "main",
    system_prompt: str | None = None,
) -> Benchmark[MathExample]:
    """Create a GSM8K benchmark.

    Rows use the standard ``question`` and ``answer`` fields. The answer field
    may contain GSM8K's worked solution and ``####`` marker.
    """

    rows = (
        examples
        if examples is not None
        else _load_huggingface(dataset, config=config, split=split)
    )
    normalized = [
        _coerce_example(
            row,
            question_keys=("question",),
            answer_keys=("answer",),
            prefix="gsm8k",
        )
        for row in rows
    ]
    return Benchmark(
        name="gsm8k",
        items=normalized,
        make_env=lambda example: GSM8KEnv(example, system_prompt=system_prompt),
        item_id=lambda example: (
            example.example_id or make_example_id("gsm8k", example.question)
        ),
        metadata={"dataset": dataset, "split": split, "task": "single_turn_math"},
    )


def aime(
    examples: Iterable[MathExample | Mapping[str, Any]] | None = None,
    *,
    split: str = "test",
    dataset: str = "MathArena/aime_2025",
    system_prompt: str | None = None,
    prompt_suffix: str = AIME_PROMPT_SUFFIX,
) -> Benchmark[MathExample]:
    """Create an AIME benchmark.

    The default dataset has ``problem`` and ``answer`` fields and contains the
    30 AIME 2025 problems. Custom rows may use either ``problem`` or
    ``question`` for the prompt.
    """

    actual_split = split
    if examples is not None:
        rows = examples
    else:
        try:
            rows = _load_huggingface(dataset, config=None, split=split)
        except ValueError as exc:
            # MathArena currently exposes AIME 2025 under ``train`` while
            # other mirrors expose it under ``test``; match Tinker Cookbook's
            # test-then-train loading behavior.
            if split != "test" or "Unknown split" not in str(exc):
                raise
            actual_split = "train"
            rows = _load_huggingface(dataset, config=None, split=actual_split)
    normalized = [
        _coerce_example(
            row,
            question_keys=("problem", "question"),
            answer_keys=("answer", "ground_truth"),
            prefix="aime",
        )
        for row in rows
    ]
    return Benchmark(
        name="aime",
        items=normalized,
        make_env=lambda example: AIMEEnv(
            example,
            system_prompt=system_prompt,
            prompt_suffix=prompt_suffix,
        ),
        item_id=lambda example: (
            example.example_id or make_example_id("aime", example.question)
        ),
        metadata={
            "dataset": dataset,
            "split": actual_split,
            "task": "single_turn_math",
        },
    )


__all__ = [
    "AIMEEnv",
    "AIME_PROMPT_SUFFIX",
    "GSM8KEnv",
    "MathEnv",
    "MathExample",
    "aime",
    "gsm8k",
    "make_example_id",
]
