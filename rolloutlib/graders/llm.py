"""Backend-neutral LLM graders."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Mapping
from typing import Any, Generic, TypeVar

from gymnasium.spaces import Space

from ..types import Chat
from .core import (
    AsyncGrader,
    Grader,
    Rubric,
    Score,
    ScoreValue,
    _with_rubric_metadata,
)


InputT = TypeVar("InputT")

Render = Callable[[InputT, Rubric | None], Chat]
Parse = Callable[[str, Rubric | None], ScoreValue]
Sample = Callable[[Chat], str]
AsyncSample = Callable[[Chat], Awaitable[str]]


def _resolve(
    response: object,
    rubric: Rubric | None,
    parse: Parse,
    metadata: Mapping[str, Any],
) -> Score:
    if not isinstance(response, str):
        raise TypeError("LLM grader sample callable must return a string")
    parsed = Score.from_value(parse(response, rubric))
    resolved_metadata = dict(metadata)
    resolved_metadata.update(parsed.metadata)
    return _with_rubric_metadata(
        Score(
            parsed.value,
            parsed.components,
            resolved_metadata,
            feedback=parsed.feedback,
        ),
        rubric,
    )


class LLMGrader(Grader[InputT], Generic[InputT]):
    """Synchronous render-sample-parse boundary for an LLM judge."""

    def __init__(
        self,
        *,
        input_space: Space[InputT],
        sample: Sample,
        render: Render[InputT],
        parse: Parse,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.input_space = input_space
        self._sample = sample
        self._render = render
        self._parse = parse
        self._metadata = dict(metadata or {})

    def _grade(
        self,
        input: InputT,
        *,
        rubric: Rubric | None = None,
    ) -> Score:
        messages = self._render(input, rubric)
        response = self._sample(messages)
        if inspect.isawaitable(response):
            close = getattr(response, "close", None)
            if callable(close):
                close()
            raise TypeError(
                "synchronous LLM grader received an awaitable; use AsyncLLMGrader"
            )
        return _resolve(
            response,
            rubric,
            self._parse,
            self._metadata,
        )


class AsyncLLMGrader(AsyncGrader[InputT], Generic[InputT]):
    """Asynchronous render-sample-parse boundary for an LLM judge."""

    def __init__(
        self,
        *,
        input_space: Space[InputT],
        sample: AsyncSample,
        render: Render[InputT],
        parse: Parse,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.input_space = input_space
        self._sample = sample
        self._render = render
        self._parse = parse
        self._metadata = dict(metadata or {})

    async def _grade(
        self,
        input: InputT,
        *,
        rubric: Rubric | None = None,
    ) -> Score:
        messages = self._render(input, rubric)
        return _resolve(
            await self._sample(messages),
            rubric,
            self._parse,
            self._metadata,
        )


__all__ = ["AsyncLLMGrader", "LLMGrader"]
