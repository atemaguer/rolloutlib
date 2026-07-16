"""Backend-neutral LLM grading."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Generator, Mapping
from typing import Any, Generic, TypeVar, cast

from ..types import Chat
from .core import Rubric, Score, ScoreValue


InputT = TypeVar("InputT")
RubricT = TypeVar("RubricT")

_Render = Callable[[InputT, RubricT], Chat]
_Sample = Callable[[Chat], str | Awaitable[str]]
_Parse = Callable[[str, RubricT], ScoreValue]


class _ParsedScore(Awaitable[Score], Generic[RubricT]):
    """Await a model response and parse it without leaking inner coroutines."""

    def __init__(
        self,
        response: Awaitable[str],
        rubric: RubricT,
        resolve: Callable[[object, RubricT], Score],
    ) -> None:
        self._response = response
        self._rubric = rubric
        self._resolve = resolve

    def __await__(self) -> Generator[Any, None, Score]:
        async def run() -> Score:
            return self._resolve(await self._response, self._rubric)

        return cast(Generator[Any, None, Score], run().__await__())

    def close(self) -> None:
        """Close an unconsumed sampler coroutine when supported."""

        close = getattr(self._response, "close", None)
        if callable(close):
            close()


class LLMGrader(Generic[InputT, RubricT]):
    """Apply a rubric through user-provided render, sample, and parse callables.

    The ``sample`` callable is the only model boundary. A closure can adapt any
    hosted API, local inference engine, or SDK into a string-producing function.
    It may be synchronous or asynchronous.
    """

    def __init__(
        self,
        *,
        sample: _Sample,
        render: _Render[InputT, RubricT],
        parse: _Parse[RubricT],
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self._sample = sample
        self._render = render
        self._parse = parse
        self._metadata = dict(metadata or {})

    def __call__(
        self,
        input: InputT,
        rubric: RubricT,
        /,
    ) -> Score | Awaitable[Score]:
        messages = self._render(input, rubric)
        sampled = self._sample(messages)
        if inspect.isawaitable(sampled):
            return _ParsedScore(sampled, rubric, self._resolve)
        return self._resolve(sampled, rubric)

    def score(self, input: InputT, rubric: RubricT, /) -> Score:
        """Run a synchronous model sampler and return its parsed score."""

        messages = self._render(input, rubric)
        sampled = self._sample(messages)
        if inspect.isawaitable(sampled):
            close = getattr(sampled, "close", None)
            if callable(close):
                close()
            raise TypeError("LLM grader returned an awaitable; use ascore()")
        return self._resolve(sampled, rubric)

    async def ascore(self, input: InputT, rubric: RubricT, /) -> Score:
        """Run a synchronous or asynchronous model sampler."""

        value = self(input, rubric)
        return await value if inspect.isawaitable(value) else value

    def _resolve(self, response: object, rubric: RubricT) -> Score:
        if not isinstance(response, str):
            raise TypeError("LLM grader sample callable must return a string")
        parsed = Score.from_value(self._parse(response, rubric))
        metadata = dict(self._metadata)
        metadata.update(parsed.metadata)
        if isinstance(rubric, Rubric):
            metadata.setdefault("rubric_fingerprint", rubric.fingerprint)
            if rubric.id is not None:
                metadata.setdefault("rubric_id", rubric.id)
        return Score(
            parsed.value,
            parsed.components,
            metadata,
            feedback=parsed.feedback,
        )


__all__ = ["LLMGrader"]
