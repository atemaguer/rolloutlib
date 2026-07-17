"""Rollout collection and trajectory data.

The rollout layer records environment interactions but does not own model
sampling. Callers provide a policy callable, which may return either a raw
action or a :class:`PolicyOutput` containing model-side sampling metadata.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Generic, Protocol, TypeVar, cast

import gymnasium as gym

from ..envs import AsyncEnv
from ..graders import Score


ObservationT = TypeVar("ObservationT")
ActionT = TypeVar("ActionT")
ItemT = TypeVar("ItemT")
PolicyObservationT = TypeVar("PolicyObservationT", contravariant=True)
PolicyActionT = TypeVar("PolicyActionT", covariant=True)


@dataclass(frozen=True, slots=True)
class PolicyOutput(Generic[ActionT]):
    """A semantic action plus sampled-token information.

    ``tokens`` and ``logprobs`` describe the generated completion, not the
    prompt. ``info`` remains available for renderer- or backend-specific data.
    The second positional argument remains ``info`` for compatibility; the
    structured sampling fields are keyword-only.
    """

    action: ActionT
    info: Mapping[str, Any] = field(default_factory=dict)
    tokens: Sequence[int] | None = field(default=None, kw_only=True)
    logprobs: Sequence[float] | None = field(default=None, kw_only=True)
    stop_reason: str | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        """Normalize sampling metadata and expose it in consistent types.

        Returns:
            ``None``. The dataclass fields are normalized during
            initialization.
        """
        info = dict(self.info)
        tokens = self.tokens
        if tokens is None and isinstance(info.get("tokens"), Sequence) and not isinstance(
            info["tokens"], (str, bytes)
        ):
            tokens = cast(Sequence[int], info["tokens"])
        logprobs = self.logprobs
        if logprobs is None and isinstance(info.get("logprobs"), Sequence) and not isinstance(
            info["logprobs"], (str, bytes)
        ):
            logprobs = cast(Sequence[float], info["logprobs"])
        stop_reason = self.stop_reason
        if stop_reason is None and isinstance(info.get("stop_reason"), str):
            stop_reason = info["stop_reason"]
        object.__setattr__(self, "info", info)
        object.__setattr__(self, "tokens", tuple(tokens) if tokens is not None else None)
        object.__setattr__(
            self,
            "logprobs",
            tuple(float(value) for value in logprobs) if logprobs is not None else None,
        )
        object.__setattr__(self, "stop_reason", stop_reason)


class Policy(Protocol[PolicyObservationT, PolicyActionT]):
    """Synchronous mapping from an observation to an environment action."""

    def __call__(
        self, observation: PolicyObservationT, /
    ) -> PolicyActionT | PolicyOutput[PolicyActionT]:
        """Map one observation to an action or structured policy output.

        Args:
            observation: Observation supplied by the environment.

        Returns:
            An environment action or a :class:`PolicyOutput` containing one.
        """
        ...


class AsyncPolicy(Protocol[PolicyObservationT, PolicyActionT]):
    """Async-compatible policy mapping with optional awaitable output."""

    def __call__(
        self, observation: PolicyObservationT, /
    ) -> (
        PolicyActionT
        | PolicyOutput[PolicyActionT]
        | Awaitable[PolicyActionT | PolicyOutput[PolicyActionT]]
    ):
        """Map one observation to an action, optionally asynchronously.

        Args:
            observation: Observation supplied by the environment.

        Returns:
            An action, :class:`PolicyOutput`, or awaitable resolving to one.
        """
        ...


@dataclass(frozen=True, slots=True)
class Step(Generic[ObservationT, ActionT]):
    """One environment interaction and its policy-side metadata."""

    observation: ObservationT
    action: ActionT
    reward: float
    next_observation: ObservationT
    terminated: bool
    truncated: bool
    info: Mapping[str, Any] = field(default_factory=dict)
    policy_info: Mapping[str, Any] = field(default_factory=dict)
    policy_tokens: Sequence[int] | None = None
    policy_logprobs: Sequence[float] | None = None
    policy_stop_reason: str | None = None

    def __post_init__(self) -> None:
        """Normalize reward and policy metadata for serialization.

        Returns:
            ``None``. The dataclass fields are normalized during
            initialization.
        """
        object.__setattr__(self, "reward", float(self.reward))
        object.__setattr__(self, "info", dict(self.info))
        object.__setattr__(self, "policy_info", dict(self.policy_info))
        object.__setattr__(
            self,
            "policy_tokens",
            tuple(self.policy_tokens) if self.policy_tokens is not None else None,
        )
        object.__setattr__(
            self,
            "policy_logprobs",
            (
                tuple(float(value) for value in self.policy_logprobs)
                if self.policy_logprobs is not None
                else None
            ),
        )

    @property
    def score(self) -> Score | None:
        """Return the structured score produced by this environment step.

        Returns:
            A :class:`Score` from ``info``, or ``None`` when no score exists.
        """

        return Score.from_info(self.info)


@dataclass(frozen=True, slots=True)
class Trajectory(Generic[ObservationT, ActionT]):
    """An ordered sequence of steps from one environment episode."""

    initial_observation: ObservationT
    steps: Sequence[Step[ObservationT, ActionT]] = ()
    initial_info: Mapping[str, Any] = field(default_factory=dict)
    terminated: bool = False
    truncated: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize steps, metadata, and episode flags.

        Returns:
            ``None``. The dataclass fields are normalized during
            initialization.
        """
        object.__setattr__(self, "steps", tuple(self.steps))
        object.__setattr__(self, "initial_info", dict(self.initial_info))
        object.__setattr__(self, "metadata", dict(self.metadata))
        object.__setattr__(self, "terminated", bool(self.terminated))
        object.__setattr__(self, "truncated", bool(self.truncated))

    def __len__(self) -> int:
        """Return the number of environment steps in the trajectory.

        Returns:
            The number of recorded steps as an integer.
        """
        return len(self.steps)

    def __iter__(self) -> Iterator[Step[ObservationT, ActionT]]:
        """Return an iterator over recorded environment steps.

        Returns:
            An iterator yielding steps in collection order.
        """
        return iter(self.steps)

    @property
    def observations(self) -> tuple[ObservationT, ...]:
        """Return the initial observation followed by each next observation.

        Returns:
            A tuple containing one more observation than the number of steps.
        """
        return (self.initial_observation,) + tuple(
            step.next_observation for step in self.steps
        )

    @property
    def actions(self) -> tuple[ActionT, ...]:
        """Return actions in the order they were applied.

        Returns:
            A tuple of actions recorded in the trajectory.
        """
        return tuple(step.action for step in self.steps)

    @property
    def rewards(self) -> tuple[float, ...]:
        """Return environment rewards in step order.

        Returns:
            A tuple of scalar rewards recorded in the trajectory.
        """
        return tuple(step.reward for step in self.steps)

    @property
    def total_reward(self) -> float:
        """Return the undiscounted sum of environment step rewards.

        Returns:
            The scalar episode return.
        """

        return sum(self.rewards)

    @property
    def score(self) -> Score:
        """Return the terminal score or a score derived from episode return.

        Returns:
            The final step's structured score when available, otherwise a
            :class:`Score` containing :attr:`total_reward`.
        """

        if self.steps:
            score = self.steps[-1].score
            if score is not None:
                return score
        return Score(value=self.total_reward)

    @property
    def complete(self) -> bool:
        """Return whether collection ended through termination or truncation.

        Returns:
            ``True`` when the trajectory terminated or was truncated.
        """

        return self.terminated or self.truncated


@dataclass(frozen=True, slots=True)
class RolloutError:
    """A serializable error captured while collecting a trajectory."""

    error_type: str
    error_message: str


@dataclass(frozen=True, slots=True)
class TrajectoryGroup(Generic[ItemT, ObservationT, ActionT]):
    """Trajectories collected for one source item."""

    item: ItemT
    item_id: str
    trajectories: Sequence[Trajectory[ObservationT, ActionT]]
    scores: Sequence[Score] = ()
    errors: Sequence[RolloutError] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize trajectories, scores, errors, and metadata.

        Returns:
            ``None``. Missing scores are inferred from the trajectories.

        Raises:
            ValueError: If the number of scores differs from trajectories.
        """
        trajectories = tuple(self.trajectories)
        scores = tuple(self.scores)
        if not scores:
            scores = tuple(trajectory.score for trajectory in trajectories)
        if len(scores) != len(trajectories):
            raise ValueError("scores must have one entry per trajectory")
        object.__setattr__(self, "trajectories", trajectories)
        object.__setattr__(self, "scores", scores)
        object.__setattr__(self, "errors", tuple(self.errors))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def rewards(self) -> tuple[float, ...]:
        """Return scalar scores suitable for consumption by training code.

        Returns:
            A tuple containing one scalar value per trajectory score.
        """

        return tuple(score.value for score in self.scores)


def _normalize_policy_output(value: object) -> PolicyOutput[Any]:
    """Convert a raw policy result into a :class:`PolicyOutput`.

    Args:
        value: Raw action or already-structured policy result.

    Returns:
        A normalized :class:`PolicyOutput` instance.
    """
    if isinstance(value, PolicyOutput):
        return value
    return PolicyOutput(action=value)


def _sync_policy_output(value: object) -> PolicyOutput[Any]:
    """Validate and normalize the result returned by a sync policy.

    Args:
        value: Value returned by a synchronous policy callable.

    Returns:
        A normalized :class:`PolicyOutput` instance.

    Raises:
        TypeError: If ``value`` is awaitable rather than an immediate result.
    """
    if inspect.isawaitable(value):
        close = getattr(value, "close", None)
        if callable(close):
            close()
        raise TypeError("synchronous policies must return an action")
    return _normalize_policy_output(value)


def _policy_info(output: PolicyOutput[Any]) -> dict[str, Any]:
    """Combine structured policy metadata into a serializable info mapping.

    Args:
        output: Normalized policy output containing optional sampling fields.

    Returns:
        A dictionary containing explicit and structured policy metadata.
    """
    info = dict(output.info)
    if output.tokens is not None:
        info.setdefault("tokens", output.tokens)
    if output.logprobs is not None:
        info.setdefault("logprobs", output.logprobs)
    if output.stop_reason is not None:
        info.setdefault("stop_reason", output.stop_reason)
    return info


def rollout(
    environment: gym.Env[ObservationT, ActionT],
    policy: Policy[ObservationT, ActionT],
    *,
    seed: int | None = None,
    options: dict[str, Any] | None = None,
    max_steps: int | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Trajectory[ObservationT, ActionT]:
    """Collect one synchronous trajectory without closing ``environment``.

    Args:
        environment: Gymnasium-compatible environment to interact with.
        policy: Synchronous callable mapping observations to actions.
        seed: Optional reset seed passed to the environment.
        options: Optional reset options passed to the environment.
        max_steps: Optional maximum number of actions before truncation.
        metadata: Optional metadata attached to the returned trajectory.

    Returns:
        The collected trajectory, including all steps and final flags.

    Raises:
        ValueError: If ``max_steps`` is negative.
        TypeError: If the policy returns an awaitable.
    """

    if max_steps is not None and max_steps < 0:
        raise ValueError("max_steps must be non-negative")
    observation, initial_info = environment.reset(seed=seed, options=options)
    steps: list[Step[ObservationT, ActionT]] = []
    terminated = False
    truncated = False
    while not (terminated or truncated):
        if max_steps is not None and len(steps) >= max_steps:
            truncated = True
            break
        output = _sync_policy_output(policy(observation))
        next_observation, reward, terminated, truncated, info = environment.step(
            cast(ActionT, output.action)
        )
        steps.append(
            Step(
                observation=observation,
                action=cast(ActionT, output.action),
                reward=float(reward),
                next_observation=next_observation,
                terminated=terminated,
                truncated=truncated,
                info=info,
                policy_info=_policy_info(output),
                policy_tokens=output.tokens,
                policy_logprobs=output.logprobs,
                policy_stop_reason=output.stop_reason,
            )
        )
        observation = next_observation
    return Trajectory(
        initial_observation=observation if not steps else steps[0].observation,
        steps=steps,
        initial_info=initial_info,
        terminated=terminated,
        truncated=truncated,
        metadata=metadata or {},
    )


async def arollout(
    environment: AsyncEnv[ObservationT, ActionT],
    policy: AsyncPolicy[ObservationT, ActionT],
    *,
    seed: int | None = None,
    options: dict[str, Any] | None = None,
    max_steps: int | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Trajectory[ObservationT, ActionT]:
    """Collect one asynchronous trajectory without closing ``environment``.

    Args:
        environment: Asynchronous environment to interact with.
        policy: Async-compatible callable mapping observations to actions.
        seed: Optional reset seed passed to the environment.
        options: Optional reset options passed to the environment.
        max_steps: Optional maximum number of actions before truncation.
        metadata: Optional metadata attached to the returned trajectory.

    Returns:
        The collected trajectory, including all steps and final flags.

    Raises:
        ValueError: If ``max_steps`` is negative.
    """

    if max_steps is not None and max_steps < 0:
        raise ValueError("max_steps must be non-negative")
    observation, initial_info = await environment.reset(seed=seed, options=options)
    steps: list[Step[ObservationT, ActionT]] = []
    terminated = False
    truncated = False
    while not (terminated or truncated):
        if max_steps is not None and len(steps) >= max_steps:
            truncated = True
            break
        value = policy(observation)
        if inspect.isawaitable(value):
            value = await value
        output = _normalize_policy_output(value)
        next_observation, reward, terminated, truncated, info = await environment.step(
            cast(ActionT, output.action)
        )
        steps.append(
            Step(
                observation=observation,
                action=cast(ActionT, output.action),
                reward=reward,
                next_observation=next_observation,
                terminated=terminated,
                truncated=truncated,
                info=info,
                policy_info=_policy_info(output),
                policy_tokens=output.tokens,
                policy_logprobs=output.logprobs,
                policy_stop_reason=output.stop_reason,
            )
        )
        observation = next_observation
    return Trajectory(
        initial_observation=observation if not steps else steps[0].observation,
        steps=steps,
        initial_info=initial_info,
        terminated=terminated,
        truncated=truncated,
        metadata=metadata or {},
    )


def rollout_group(
    item: ItemT,
    make_env: Callable[[ItemT], gym.Env[ObservationT, ActionT]],
    policy: Policy[ObservationT, ActionT],
    *,
    num_rollouts: int = 1,
    item_id: str | None = None,
    max_steps: int | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> TrajectoryGroup[ItemT, ObservationT, ActionT]:
    """Collect independent synchronous trajectories for one item.

    Args:
        item: Dataset item used to create each environment.
        make_env: Factory returning a fresh environment for ``item``.
        policy: Synchronous callable mapping observations to actions.
        num_rollouts: Number of independent trajectories to collect.
        item_id: Optional stable identifier for the item.
        max_steps: Optional per-trajectory action limit.
        metadata: Optional metadata attached to the trajectory group.

    Returns:
        A trajectory group containing the requested rollouts and scores.

    Raises:
        ValueError: If ``num_rollouts`` is less than one.
    """

    if num_rollouts < 1:
        raise ValueError("num_rollouts must be at least 1")
    trajectories: list[Trajectory[ObservationT, ActionT]] = []
    try:
        for _ in range(num_rollouts):
            environment = make_env(item)
            try:
                trajectories.append(rollout(environment, policy, max_steps=max_steps))
            finally:
                environment.close()
    except Exception:
        raise
    return TrajectoryGroup(
        item=item,
        item_id=item_id or "0",
        trajectories=trajectories,
        metadata=metadata or {},
    )


async def arollout_group(
    item: ItemT,
    make_env: Callable[[ItemT], AsyncEnv[ObservationT, ActionT] | Awaitable[AsyncEnv[ObservationT, ActionT]]],
    policy: AsyncPolicy[ObservationT, ActionT],
    *,
    num_rollouts: int = 1,
    item_id: str | None = None,
    max_steps: int | None = None,
    concurrency: int = 1,
    metadata: Mapping[str, Any] | None = None,
) -> TrajectoryGroup[ItemT, ObservationT, ActionT]:
    """Collect asynchronous trajectories for one item with bounded concurrency.

    Args:
        item: Dataset item used to create each environment.
        make_env: Sync or async factory returning a fresh async environment.
        policy: Async-compatible callable mapping observations to actions.
        num_rollouts: Number of independent trajectories to collect.
        item_id: Optional stable identifier for the item.
        max_steps: Optional per-trajectory action limit.
        concurrency: Maximum number of environments collected concurrently.
        metadata: Optional metadata attached to the trajectory group.

    Returns:
        A trajectory group containing the requested rollouts and scores.

    Raises:
        ValueError: If ``num_rollouts`` or ``concurrency`` is less than one.
    """

    if num_rollouts < 1:
        raise ValueError("num_rollouts must be at least 1")
    if concurrency < 1:
        raise ValueError("concurrency must be at least 1")
    semaphore = asyncio.Semaphore(concurrency)

    async def collect_one() -> Trajectory[ObservationT, ActionT]:
        """Collect one trajectory while holding a concurrency permit.

        Returns:
            A trajectory collected from a fresh environment instance.
        """
        async with semaphore:
            environment = make_env(item)
            if inspect.isawaitable(environment):
                environment = await environment
            try:
                return await arollout(environment, policy, max_steps=max_steps)
            finally:
                await environment.close()

    trajectories = list(
        await asyncio.gather(*(collect_one() for _ in range(num_rollouts)))
    )
    return TrajectoryGroup(
        item=item,
        item_id=item_id or "0",
        trajectories=trajectories,
        metadata=metadata or {},
    )


__all__ = [
    "ActionT",
    "AsyncPolicy",
    "ItemT",
    "ObservationT",
    "Policy",
    "PolicyOutput",
    "RolloutError",
    "Step",
    "Trajectory",
    "TrajectoryGroup",
    "arollout",
    "arollout_group",
    "rollout",
    "rollout_group",
]
