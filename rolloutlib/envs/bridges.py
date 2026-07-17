"""Bridges between synchronous and asynchronous environment conventions."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable
from functools import partial
from typing import Any, Coroutine, TypeVar, overload

import gymnasium as gym

from .core import AsyncEnv


ObsT = TypeVar("ObsT")
ActT = TypeVar("ActT")
ResultT = TypeVar("ResultT")


class AsyncFromSync(AsyncEnv[ObsT, ActT]):
    """Lift an existing Gymnasium environment into the async convention.

    Synchronous calls run in a worker thread so they do not block the event loop.
    One lock serializes reset, step, and close for each wrapped environment.
    """

    def __init__(self, env: gym.Env[ObsT, ActT]) -> None:
        """Create an asynchronous bridge around a Gymnasium environment.

        Args:
            env: Synchronous environment to execute in a worker thread.

        Returns:
            ``None``.
        """
        if not isinstance(env, gym.Env):
            raise TypeError(f"expected gymnasium.Env, got {type(env).__name__}")
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.metadata = env.metadata
        self._call_lock = asyncio.Lock()
        self._closed = False

    async def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsT, dict[str, Any]]:
        """Reset the wrapped environment without blocking the event loop.

        Args:
            seed: Optional reset seed.
            options: Optional application-defined reset options.

        Returns:
            The wrapped environment's initial observation and info dictionary.
        """
        async with self._call_lock:
            self._ensure_open()
            return await self._run_sync(
                partial(self.env.reset, seed=seed, options=options)
            )

    async def step(
        self, action: ActT
    ) -> tuple[ObsT, float, bool, bool, dict[str, Any]]:
        """Advance the wrapped environment asynchronously.

        Args:
            action: Action passed to the wrapped environment.

        Returns:
            The standard Gymnasium five-tuple.
        """
        async with self._call_lock:
            self._ensure_open()
            observation, reward, terminated, truncated, info = await self._run_sync(
                partial(self.env.step, action)
            )
            return observation, float(reward), terminated, truncated, info

    async def close(self) -> None:
        """Close the wrapped environment exactly once.

        Returns:
            ``None``.
        """
        async with self._call_lock:
            if self._closed:
                return
            await self._run_sync(self.env.close)
            self._closed = True

    async def _run_sync(self, call: Callable[[], ResultT]) -> ResultT:
        """Run a synchronous callable in a worker thread.

        Args:
            call: Zero-argument callable to execute.

        Returns:
            The callable's return value.
        """
        task = asyncio.create_task(asyncio.to_thread(call))
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            # The worker thread cannot be cancelled. Wait for it before allowing a
            # subsequent stateful call to acquire this bridge's lock.
            while not task.done():
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    # Repeated cancellation still must not permit overlap.
                    continue
                except BaseException:
                    # The original coroutine cancellation remains authoritative,
                    # but retrieving the worker exception prevents a task warning.
                    break
            if task.done() and not task.cancelled():
                task.exception()
            raise

    def _ensure_open(self) -> None:
        """Raise ``RuntimeError`` if the wrapped environment is closed.

        Returns:
            ``None``.
        """
        if self._closed:
            raise RuntimeError("environment is closed")


class SyncFromAsync(gym.Env[ObsT, ActT]):
    """Expose an async environment through the synchronous Gymnasium API.

    The bridge owns one persistent event loop in a background thread. Async
    resources used by the environment should be created on that loop (typically
    lazily during ``reset``) and remain there for the bridge's lifetime.
    """

    def __init__(self, env: AsyncEnv[ObsT, ActT]) -> None:
        """Create a synchronous bridge around an async environment.

        Args:
            env: Asynchronous environment to run on a background event loop.

        Returns:
            ``None``.
        """
        super().__init__()
        if not isinstance(env, AsyncEnv):
            raise TypeError(f"expected AsyncEnv, got {type(env).__name__}")
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.metadata = env.metadata
        self._call_lock = threading.Lock()
        self._closed = False
        self._loop = asyncio.new_event_loop()
        self._loop_ready = threading.Event()
        self._loop_thread = threading.Thread(
            target=self._run_event_loop,
            name=f"{type(self).__name__}-{id(self):x}",
            daemon=True,
        )
        self._loop_thread.start()
        self._loop_ready.wait()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsT, dict[str, Any]]:
        """Synchronously reset the wrapped asynchronous environment.

        Args:
            seed: Optional reset seed.
            options: Optional application-defined reset options.

        Returns:
            The wrapped environment's initial observation and info dictionary.
        """
        self._ensure_not_loop_thread()
        with self._call_lock:
            self._ensure_open()
            super().reset(seed=seed)
            return self._run_async(self.env.reset(seed=seed, options=options))

    def step(self, action: ActT) -> tuple[ObsT, float, bool, bool, dict[str, Any]]:
        """Synchronously advance the wrapped asynchronous environment.

        Args:
            action: Action passed to the wrapped environment.

        Returns:
            The standard Gymnasium five-tuple.
        """
        self._ensure_not_loop_thread()
        with self._call_lock:
            self._ensure_open()
            observation, reward, terminated, truncated, info = self._run_async(
                self.env.step(action)
            )
            return observation, float(reward), terminated, truncated, info

    def close(self) -> None:
        """Close the wrapped environment and stop its background event loop.

        Returns:
            ``None``.
        """
        self._ensure_not_loop_thread()
        should_stop = False
        try:
            with self._call_lock:
                if self._closed:
                    return
                try:
                    self._run_async(self.env.close())
                finally:
                    self._closed = True
                    should_stop = True
        finally:
            if should_stop:
                self._loop.call_soon_threadsafe(self._loop.stop)
                self._loop_thread.join()

    def _run_event_loop(self) -> None:
        """Run and then clean up the bridge's persistent event loop.

        Returns:
            ``None``.
        """
        asyncio.set_event_loop(self._loop)
        self._loop_ready.set()
        try:
            self._loop.run_forever()
        finally:
            pending = asyncio.all_tasks(self._loop)
            for task in pending:
                task.cancel()
            if pending:
                self._loop.run_until_complete(
                    asyncio.gather(*pending, return_exceptions=True)
                )
            self._loop.run_until_complete(self._loop.shutdown_asyncgens())
            self._loop.run_until_complete(self._loop.shutdown_default_executor())
            self._loop.close()

    def _run_async(self, coroutine: Coroutine[Any, Any, ResultT]) -> ResultT:
        """Execute a coroutine on the bridge's background event loop.

        Args:
            coroutine: Coroutine to execute.

        Returns:
            The coroutine's result.
        """
        future = asyncio.run_coroutine_threadsafe(coroutine, self._loop)
        try:
            return future.result()
        except BaseException:
            if future.done():
                raise
            # An interruption of the synchronous caller does not stop work on the
            # background loop. Keep this call serialized until that work finishes.
            while not future.done():
                try:
                    future.result()
                except BaseException:
                    continue
            future.exception()
            raise

    def _ensure_not_loop_thread(self) -> None:
        """Reject calls made recursively from the bridge event-loop thread.

        Returns:
            ``None``.
        """
        if threading.current_thread() is self._loop_thread:
            raise RuntimeError(
                "synchronous bridge methods cannot be called from their event-loop "
                "thread"
            )

    def _ensure_open(self) -> None:
        """Raise ``RuntimeError`` if this bridge has been closed.

        Returns:
            ``None``.
        """
        if self._closed:
            raise RuntimeError("environment is closed")

    def _detach(self) -> AsyncEnv[ObsT, ActT]:
        """Stop this bridge's loop without closing its inner environment.

        Returns:
            The still-open inner asynchronous environment.
        """
        self._ensure_not_loop_thread()
        should_stop = False
        with self._call_lock:
            if not self._closed:
                self._closed = True
                should_stop = True
        if should_stop:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._loop_thread.join()
        return self.env


@overload
def as_async(env: AsyncEnv[ObsT, ActT]) -> AsyncEnv[ObsT, ActT]:
    """Type-only overload for an already asynchronous environment.

    Args:
        env: Asynchronous environment to preserve.

    Returns:
        The same asynchronous environment type.
    """
    ...


@overload
def as_async(env: gym.Env[ObsT, ActT]) -> AsyncEnv[ObsT, ActT]:
    """Type-only overload for adapting a synchronous environment.

    Args:
        env: Synchronous Gymnasium environment to adapt.

    Returns:
        An asynchronous environment wrapper.
    """
    ...


def as_async(
    env: AsyncEnv[ObsT, ActT] | gym.Env[ObsT, ActT],
) -> AsyncEnv[ObsT, ActT]:
    """Return an environment in the asynchronous convention.

    Args:
        env: A synchronous Gymnasium or asynchronous rolloutlib environment.

    Returns:
        An ``AsyncEnv``; existing async environments are returned unchanged.
    """
    if isinstance(env, SyncFromAsync):
        return env._detach()
    if isinstance(env, AsyncEnv):
        return env
    if isinstance(env, gym.Env):
        return AsyncFromSync(env)
    raise TypeError(
        f"expected rolloutlib.envs.AsyncEnv or gymnasium.Env, got {type(env).__name__}"
    )


@overload
def as_sync(env: gym.Env[ObsT, ActT]) -> gym.Env[ObsT, ActT]:
    """Type-only overload for an already synchronous environment.

    Args:
        env: Synchronous environment to preserve.

    Returns:
        The same synchronous environment type.
    """
    ...


@overload
def as_sync(env: AsyncEnv[ObsT, ActT]) -> SyncFromAsync[ObsT, ActT]:
    """Type-only overload for adapting an asynchronous environment.

    Args:
        env: Asynchronous environment to adapt.

    Returns:
        A synchronous bridge around the asynchronous environment.
    """
    ...


def as_sync(
    env: gym.Env[ObsT, ActT] | AsyncEnv[ObsT, ActT],
) -> gym.Env[ObsT, ActT]:
    """Return an environment in the synchronous Gymnasium convention.

    Args:
        env: A synchronous Gymnasium or asynchronous rolloutlib environment.

    Returns:
        A Gymnasium environment; existing synchronous environments are returned
        unchanged.
    """
    if isinstance(env, AsyncFromSync):
        return env.env
    if isinstance(env, gym.Env):
        return env
    if isinstance(env, AsyncEnv):
        return SyncFromAsync(env)
    raise TypeError(
        f"expected gymnasium.Env or rolloutlib.envs.AsyncEnv, got {type(env).__name__}"
    )
