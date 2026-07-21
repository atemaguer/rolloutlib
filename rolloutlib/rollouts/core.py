"""Rollout collection and trajectory data.

The rollout layer records environment interactions but does not own model
sampling. Callers provide a policy callable, which may return either a raw
action or a :class:`PolicyOutput` containing model-side sampling metadata.
"""

from __future__ import annotations

import asyncio
import inspect
import math
from collections.abc import Awaitable, Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Generic, Protocol, TypeVar, cast

import gymnasium as gym
import numpy as np
from gymnasium.vector import AutoresetMode, SyncVectorEnv
from gymnasium.vector.utils import concatenate, create_empty_array, iterate

from .._awaitables import resolve
from ..envs import Env
from ..graders import Score
from ..spaces.compatibility import (
    check_space_compatibility,
    check_space_value,
    require_space,
)


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
    """A policy mapping whose result may be immediate or awaitable."""

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


class BatchPolicy(Protocol[PolicyObservationT, PolicyActionT]):
    """A batch policy whose result may be immediate or awaitable."""

    def __call__(
        self, observations: Sequence[PolicyObservationT], /
    ) -> (
        Sequence[PolicyActionT | PolicyOutput[PolicyActionT]]
        | Awaitable[Sequence[PolicyActionT | PolicyOutput[PolicyActionT]]]
    ):
        """Return exactly one action or structured output per observation."""
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
    """Trajectories collected for one input item."""

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


def _batch_policy_outputs(value: object, *, expected: int) -> tuple[PolicyOutput[Any], ...]:
    """Normalize one policy result per observation in a batch."""
    if inspect.isawaitable(value):
        close = getattr(value, "close", None)
        if callable(close):
            close()
        raise TypeError("synchronous batch policies must return actions")
    if isinstance(value, (str, bytes, Mapping)):
        raise TypeError("batch policy must return a sequence of actions")
    try:
        outputs = tuple(_normalize_policy_output(item) for item in cast(Any, value))
    except TypeError as error:
        raise TypeError("batch policy must return a sequence of actions") from error
    if len(outputs) != expected:
        raise ValueError(
            f"batch policy returned {len(outputs)} outputs for {expected} observations"
        )
    return outputs


def _split_vector_values(
    batch_space: gym.Space[Any],
    values: object,
) -> tuple[Any, ...]:
    """Split a Gymnasium vector-space value into its per-environment values."""
    try:
        return tuple(iterate(batch_space, values))
    except (TypeError, ValueError) as error:
        raise TypeError("vector environment returned an invalid batched value") from error


def _split_vector_info(
    info: Mapping[str, Any],
    num_envs: int,
) -> tuple[dict[str, Any], ...]:
    """Convert Gymnasium's masked vector-info mapping into per-environment info."""
    result = [dict() for _ in range(num_envs)]
    for key, value in info.items():
        if key.startswith("_"):
            continue
        mask_value = info.get(f"_{key}", np.ones(num_envs, dtype=np.bool_))
        mask = np.asarray(mask_value, dtype=np.bool_)
        if mask.shape != (num_envs,):
            raise ValueError(f"vector info mask for {key!r} must have shape ({num_envs},)")
        if isinstance(value, Mapping):
            nested = _split_vector_info(value, num_envs)
            for index in range(num_envs):
                if mask[index]:
                    result[index][key] = nested[index]
            continue
        try:
            if len(value) != num_envs:
                raise ValueError
        except (TypeError, ValueError):
            raise ValueError(
                f"vector info value for {key!r} must have one entry per environment"
            ) from None
        for index in range(num_envs):
            if mask[index]:
                result[index][key] = value[index]
    return tuple(result)


def _check_composition(
    environment: Any,
    policy: object,
) -> None:
    """Validate declared environment and policy spaces before a rollout."""

    try:
        observation_space = require_space(
            environment.observation_space,
            name="environment observation_space",
        )
        action_space = require_space(
            environment.action_space,
            name="environment action_space",
        )
    except AttributeError as error:
        raise TypeError(
            "environment must define action_space and observation_space"
        ) from error

    policy_observation_space = getattr(policy, "observation_space", None)
    if policy_observation_space is not None:
        policy_observation_space = require_space(
            policy_observation_space,
            name="policy observation_space",
        )
        check_space_compatibility(
            observation_space,
            policy_observation_space,
            produced_name="environment observation_space",
            accepted_name="policy observation_space",
        )

    policy_action_space = getattr(policy, "action_space", None)
    if policy_action_space is not None:
        policy_action_space = require_space(
            policy_action_space,
            name="policy action_space",
        )
        check_space_compatibility(
            policy_action_space,
            action_space,
            produced_name="policy action_space",
            accepted_name="environment action_space",
        )


def _check_reset_result(
    environment: Any,
    observation: object,
    info: object,
) -> None:
    check_space_value(
        environment.observation_space,
        observation,
        name="reset observation",
    )
    if not isinstance(info, dict):
        raise TypeError("environment reset info must be a dictionary")


def _check_step_result(
    environment: Any,
    observation: object,
    reward: object,
    terminated: object,
    truncated: object,
    info: object,
) -> None:
    check_space_value(
        environment.observation_space,
        observation,
        name="step observation",
    )
    if isinstance(reward, bool):
        raise TypeError("environment reward must be a finite number")
    try:
        scalar_reward = float(cast(Any, reward))
    except (TypeError, ValueError) as error:
        raise TypeError("environment reward must be a finite number") from error
    if not math.isfinite(scalar_reward):
        raise ValueError("environment reward must be finite")
    if not isinstance(terminated, (bool, np.bool_)) or not isinstance(
        truncated, (bool, np.bool_)
    ):
        raise TypeError("environment termination flags must be booleans")
    if not isinstance(info, dict):
        raise TypeError("environment step info must be a dictionary")


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
    _check_composition(environment, policy)
    observation, initial_info = environment.reset(seed=seed, options=options)
    _check_reset_result(environment, observation, initial_info)
    steps: list[Step[ObservationT, ActionT]] = []
    terminated = False
    truncated = False
    while not (terminated or truncated):
        if max_steps is not None and len(steps) >= max_steps:
            truncated = True
            break
        output = _sync_policy_output(policy(observation))
        check_space_value(
            environment.action_space,
            output.action,
            name="policy action",
        )
        next_observation, reward, terminated, truncated, info = environment.step(
            cast(ActionT, output.action)
        )
        _check_step_result(
            environment,
            next_observation,
            reward,
            terminated,
            truncated,
            info,
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
    environment: Env[ObservationT, ActionT],
    policy: Policy[ObservationT, ActionT],
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
    _check_composition(environment, policy)
    observation, initial_info = await resolve(
        environment.reset(seed=seed, options=options)
    )
    _check_reset_result(environment, observation, initial_info)
    steps: list[Step[ObservationT, ActionT]] = []
    terminated = False
    truncated = False
    while not (terminated or truncated):
        if max_steps is not None and len(steps) >= max_steps:
            truncated = True
            break
        output = _normalize_policy_output(await resolve(policy(observation)))
        check_space_value(
            environment.action_space,
            output.action,
            name="policy action",
        )
        next_observation, reward, terminated, truncated, info = await resolve(
            environment.step(cast(ActionT, output.action))
        )
        _check_step_result(
            environment,
            next_observation,
            reward,
            terminated,
            truncated,
            info,
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


def _rollout_group_scalar(
    item: ItemT,
    make_env: Callable[[ItemT], gym.Env[ObservationT, ActionT]],
    policy: Policy[ObservationT, ActionT],
    *,
    num_rollouts: int = 1,
    item_id: str | None = None,
    seed: int | Sequence[int] | None = None,
    options: dict[str, Any] | None = None,
    max_steps: int | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> TrajectoryGroup[ItemT, ObservationT, ActionT]:
    """Collect independent synchronous trajectories for one item.

    Args:
        item: Input item used to create each environment.
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
    if isinstance(seed, Sequence) and not isinstance(seed, (str, bytes)):
        seeds: list[int | None] = list(seed)
        if len(seeds) != num_rollouts:
            raise ValueError("seed sequence must have one entry per rollout")
    elif isinstance(seed, int):
        seeds = [seed + index for index in range(num_rollouts)]
    else:
        seeds = [None] * num_rollouts
    trajectories: list[Trajectory[ObservationT, ActionT]] = []
    try:
        for index in range(num_rollouts):
            environment = make_env(item)
            try:
                trajectories.append(
                    rollout(
                        environment,
                        policy,
                        seed=seeds[index],
                        options=options,
                        max_steps=max_steps,
                    )
                )
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


def _rollout_group_batch(
    item: ItemT,
    make_env: Callable[[ItemT], gym.Env[ObservationT, ActionT]],
    policy: BatchPolicy[ObservationT, ActionT],
    *,
    num_rollouts: int = 1,
    item_id: str | None = None,
    seed: int | Sequence[int] | None = None,
    options: dict[str, Any] | None = None,
    max_steps: int | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> TrajectoryGroup[ItemT, ObservationT, ActionT]:
    """Collect a synchronous batch, beginning through ``SyncVectorEnv``.

    The collector makes one policy call per active wave. If vector slots finish
    at different times, it continues the unfinished underlying environments
    directly rather than requiring users to select another public function.

    Args:
        item: Input item used to create every environment in the group.
        make_env: Factory returning a fresh Gymnasium environment for ``item``.
        policy: Callable mapping a sequence of observations to equally many
            actions or :class:`PolicyOutput` instances.
        num_rollouts: Number of vector-environment slots and trajectories.
        item_id: Optional stable identifier for the item.
        seed: Optional scalar seed or one seed per vector slot.
        options: Optional reset options passed to every environment.
        max_steps: Optional shared action limit.
        metadata: Optional metadata attached to the trajectory group.

    Returns:
        A trajectory group in vector-slot order.

    Raises:
        ValueError: If counts, limits, policy outputs, or Gymnasium values are
            invalid.
    """
    if num_rollouts < 1:
        raise ValueError("num_rollouts must be at least 1")
    if max_steps is not None and max_steps < 0:
        raise ValueError("max_steps must be non-negative")

    environments: list[gym.Env[ObservationT, ActionT]] = []
    vector_environment: SyncVectorEnv | None = None
    try:
        for _ in range(num_rollouts):
            environments.append(make_env(item))
        vector_environment = SyncVectorEnv(
            [lambda environment=environment: environment for environment in environments],
            autoreset_mode=AutoresetMode.DISABLED,
        )
        for environment in environments:
            _check_composition(environment, policy)

        reset_seed = (
            list(seed)
            if isinstance(seed, Sequence) and not isinstance(seed, (str, bytes))
            else seed
        )
        batched_observations, batched_initial_info = vector_environment.reset(
            seed=cast(int | list[int | None] | None, reset_seed),
            options=options,
        )
        observations = list(
            cast(
                tuple[ObservationT, ...],
                _split_vector_values(
                    vector_environment.observation_space,
                    batched_observations,
                ),
            )
        )
        if not isinstance(batched_initial_info, Mapping):
            raise TypeError("vector environment reset info must be a mapping")
        initial_infos = _split_vector_info(batched_initial_info, num_rollouts)
        for environment, observation, info in zip(
            environments,
            observations,
            initial_infos,
            strict=True,
        ):
            _check_reset_result(environment, observation, info)

        initial_observations = tuple(observations)
        steps: list[list[Step[ObservationT, ActionT]]] = [
            [] for _ in range(num_rollouts)
        ]
        terminations = [False] * num_rollouts
        truncations = [False] * num_rollouts

        vector_mode = True
        while True:
            for index in range(num_rollouts):
                if (
                    not terminations[index]
                    and not truncations[index]
                    and max_steps is not None
                    and len(steps[index]) >= max_steps
                ):
                    truncations[index] = True
            active = [
                index
                for index in range(num_rollouts)
                if not (terminations[index] or truncations[index])
            ]
            if not active:
                break

            active_observations = tuple(observations[index] for index in active)
            outputs = _batch_policy_outputs(
                policy(active_observations),
                expected=len(active),
            )
            actions: list[ActionT] = []
            for index, output in zip(active, outputs, strict=True):
                check_space_value(
                    environments[index].action_space,
                    output.action,
                    name="policy action",
                )
                actions.append(cast(ActionT, output.action))

            if vector_mode:
                batched_actions = concatenate(
                    vector_environment.single_action_space,
                    actions,
                    create_empty_array(
                        vector_environment.single_action_space,
                        n=num_rollouts,
                    ),
                )
                (
                    batched_next_observations,
                    rewards,
                    terminated_values,
                    truncated_values,
                    batched_info,
                ) = vector_environment.step(batched_actions)
                next_observations = cast(
                    tuple[ObservationT, ...],
                    _split_vector_values(
                        vector_environment.observation_space,
                        batched_next_observations,
                    ),
                )
                infos = _split_vector_info(batched_info, num_rollouts)
                step_results = tuple(
                    zip(
                        next_observations,
                        rewards,
                        terminated_values,
                        truncated_values,
                        infos,
                        strict=True,
                    )
                )
            else:
                step_results = tuple(
                    environments[index].step(action)
                    for index, action in zip(active, actions, strict=True)
                )

            for index, output, action, result in zip(
                active,
                outputs,
                actions,
                step_results,
                strict=True,
            ):
                next_observation, reward, terminated, truncated, info = result
                _check_step_result(
                    environments[index],
                    next_observation,
                    reward,
                    terminated,
                    truncated,
                    info,
                )
                steps[index].append(
                    Step(
                        observation=observations[index],
                        action=action,
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
                observations[index] = next_observation
                terminations[index] = bool(terminated)
                truncations[index] = bool(truncated)

            if vector_mode and any(
                terminated or truncated
                for terminated, truncated in zip(
                    terminations,
                    truncations,
                    strict=True,
                )
            ):
                vector_mode = False

        trajectories = [
            Trajectory(
                initial_observation=initial_observations[index],
                steps=member_steps,
                initial_info=initial_infos[index],
                terminated=terminations[index],
                truncated=truncations[index],
            )
            for index, member_steps in enumerate(steps)
        ]
    finally:
        if vector_environment is not None:
            vector_environment.close()
        else:
            for environment in environments:
                environment.close()

    return TrajectoryGroup(
        item=item,
        item_id=item_id or "0",
        trajectories=trajectories,
        metadata=metadata or {},
    )


def rollout_group(
    item: ItemT,
    make_env: Callable[[ItemT], gym.Env[ObservationT, ActionT]],
    policy: Policy[ObservationT, ActionT] | None = None,
    *,
    batch_policy: BatchPolicy[ObservationT, ActionT] | None = None,
    num_rollouts: int = 1,
    item_id: str | None = None,
    seed: int | Sequence[int] | None = None,
    options: dict[str, Any] | None = None,
    max_steps: int | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> TrajectoryGroup[ItemT, ObservationT, ActionT]:
    """Collect a synchronous trajectory group with a scalar or batch policy.

    Supply exactly one of ``policy`` and ``batch_policy``. Scalar policies are
    evaluated independently for backward compatibility. Batch policies receive
    all active observations on each wave. The batch path begins with a
    Gymnasium ``SyncVectorEnv`` and transparently continues unfinished
    environments when episode lengths differ.

    Args:
        item: Input item used to create every environment in the group.
        make_env: Factory returning a fresh Gymnasium environment for ``item``.
        policy: Optional scalar policy called once per observation.
        batch_policy: Optional policy called with all active observations.
        num_rollouts: Number of trajectories to collect.
        item_id: Optional stable identifier for the item.
        seed: Optional scalar seed or one seed per rollout.
        options: Optional reset options passed to every environment.
        max_steps: Optional per-trajectory action limit.
        metadata: Optional metadata attached to the trajectory group.

    Returns:
        A trajectory group in environment-creation order.

    Raises:
        ValueError: If exactly one policy is not supplied or an argument is
            invalid.
    """
    if (policy is None) == (batch_policy is None):
        raise ValueError("provide exactly one of policy or batch_policy")
    if batch_policy is not None:
        return _rollout_group_batch(
            item,
            make_env,
            batch_policy,
            num_rollouts=num_rollouts,
            item_id=item_id,
            seed=seed,
            options=options,
            max_steps=max_steps,
            metadata=metadata,
        )
    return _rollout_group_scalar(
        item,
        make_env,
        cast(Policy[ObservationT, ActionT], policy),
        num_rollouts=num_rollouts,
        item_id=item_id,
        seed=seed,
        options=options,
        max_steps=max_steps,
        metadata=metadata,
    )


async def _arollout_group_scalar(
    item: ItemT,
    make_env: Callable[[ItemT], Env[ObservationT, ActionT] | Awaitable[Env[ObservationT, ActionT]]],
    policy: Policy[ObservationT, ActionT],
    *,
    num_rollouts: int = 1,
    item_id: str | None = None,
    max_steps: int | None = None,
    concurrency: int = 1,
    metadata: Mapping[str, Any] | None = None,
) -> TrajectoryGroup[ItemT, ObservationT, ActionT]:
    """Collect asynchronous trajectories for one item with bounded concurrency.

    Args:
        item: Input item used to create each environment.
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
                await resolve(environment.close())

    trajectories = list(
        await asyncio.gather(*(collect_one() for _ in range(num_rollouts)))
    )
    return TrajectoryGroup(
        item=item,
        item_id=item_id or "0",
        trajectories=trajectories,
        metadata=metadata or {},
    )


async def _arollout_group_batch(
    item: ItemT,
    make_env: Callable[
        [ItemT],
        Env[ObservationT, ActionT]
        | Awaitable[Env[ObservationT, ActionT]],
    ],
    policy: BatchPolicy[ObservationT, ActionT],
    *,
    num_rollouts: int = 1,
    item_id: str | None = None,
    max_steps: int | None = None,
    concurrency: int | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> TrajectoryGroup[ItemT, ObservationT, ActionT]:
    """Collect uneven async episodes with one policy call per active wave.

    Finished environments leave the next policy batch while unfinished ones
    continue. Environment reset and step calls run concurrently, bounded by
    ``concurrency``; model sampling remains under the batch policy's control.

    Args:
        item: Input item used to create every environment in the group.
        make_env: Sync or async factory returning a fresh async environment.
        policy: Async-compatible callable returning one action per active
            observation.
        num_rollouts: Number of trajectories in the group.
        item_id: Optional stable identifier for the item.
        max_steps: Optional per-trajectory action limit.
        concurrency: Maximum concurrent environment resets or steps. ``None``
            permits all group members to run concurrently.
        metadata: Optional metadata attached to the trajectory group.

    Returns:
        A trajectory group in environment-creation order.
    """
    if num_rollouts < 1:
        raise ValueError("num_rollouts must be at least 1")
    if max_steps is not None and max_steps < 0:
        raise ValueError("max_steps must be non-negative")
    if concurrency is not None and concurrency < 1:
        raise ValueError("concurrency must be at least 1")
    semaphore = asyncio.Semaphore(concurrency or num_rollouts)

    async def bounded(awaitable: Awaitable[Any]) -> Any:
        async with semaphore:
            return await awaitable

    async def create_one() -> Env[ObservationT, ActionT]:
        async with semaphore:
            environment = make_env(item)
            if inspect.isawaitable(environment):
                environment = await environment
            return environment

    environments: list[Env[ObservationT, ActionT]] = []
    try:
        creation_results = await asyncio.gather(
            *(create_one() for _ in range(num_rollouts)),
            return_exceptions=True,
        )
        creation_error: BaseException | None = None
        for result in creation_results:
            if isinstance(result, BaseException):
                if creation_error is None:
                    creation_error = result
            else:
                environments.append(result)
        if creation_error is not None:
            raise creation_error

        for environment in environments:
            _check_composition(environment, policy)

        first_environment = environments[0]
        for index, environment in enumerate(environments[1:], start=1):
            check_space_compatibility(
                first_environment.observation_space,
                environment.observation_space,
                produced_name="environment 0 observation_space",
                accepted_name=f"environment {index} observation_space",
            )
            check_space_compatibility(
                environment.observation_space,
                first_environment.observation_space,
                produced_name=f"environment {index} observation_space",
                accepted_name="environment 0 observation_space",
            )
            check_space_compatibility(
                first_environment.action_space,
                environment.action_space,
                produced_name="environment 0 action_space",
                accepted_name=f"environment {index} action_space",
            )
            check_space_compatibility(
                environment.action_space,
                first_environment.action_space,
                produced_name=f"environment {index} action_space",
                accepted_name="environment 0 action_space",
            )

        reset_results = await asyncio.gather(
            *(bounded(resolve(environment.reset())) for environment in environments)
        )
        observations: list[ObservationT] = []
        initial_infos: list[dict[str, Any]] = []
        for environment, (observation, info) in zip(
            environments,
            reset_results,
            strict=True,
        ):
            _check_reset_result(environment, observation, info)
            observations.append(observation)
            initial_infos.append(info)
        initial_observations = list(observations)
        steps: list[list[Step[ObservationT, ActionT]]] = [
            [] for _ in range(num_rollouts)
        ]
        terminations = [False] * num_rollouts
        truncations = [False] * num_rollouts

        while True:
            for index in range(num_rollouts):
                if (
                    not terminations[index]
                    and not truncations[index]
                    and max_steps is not None
                    and len(steps[index]) >= max_steps
                ):
                    truncations[index] = True
            active = [
                index
                for index in range(num_rollouts)
                if not (terminations[index] or truncations[index])
            ]
            if not active:
                break

            active_observations = tuple(observations[index] for index in active)
            value = policy(active_observations)
            if inspect.isawaitable(value):
                value = await value
            outputs = _batch_policy_outputs(value, expected=len(active))
            actions: list[ActionT] = []
            for index, output in zip(active, outputs, strict=True):
                check_space_value(
                    environments[index].action_space,
                    output.action,
                    name="policy action",
                )
                actions.append(cast(ActionT, output.action))

            step_results = await asyncio.gather(
                *(
                    bounded(resolve(environments[index].step(action)))
                    for index, action in zip(active, actions, strict=True)
                )
            )
            for index, output, action, result in zip(
                active,
                outputs,
                actions,
                step_results,
                strict=True,
            ):
                next_observation, reward, terminated, truncated, info = result
                _check_step_result(
                    environments[index],
                    next_observation,
                    reward,
                    terminated,
                    truncated,
                    info,
                )
                steps[index].append(
                    Step(
                        observation=observations[index],
                        action=action,
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
                observations[index] = next_observation
                terminations[index] = bool(terminated)
                truncations[index] = bool(truncated)

        trajectories = [
            Trajectory(
                initial_observation=initial_observations[index],
                steps=member_steps,
                initial_info=initial_infos[index],
                terminated=terminations[index],
                truncated=truncations[index],
            )
            for index, member_steps in enumerate(steps)
        ]
    finally:
        if environments:
            await asyncio.gather(
                *(resolve(environment.close()) for environment in environments)
            )

    return TrajectoryGroup(
        item=item,
        item_id=item_id or "0",
        trajectories=trajectories,
        metadata=metadata or {},
    )


async def arollout_group(
    item: ItemT,
    make_env: Callable[
        [ItemT],
        Env[ObservationT, ActionT]
        | Awaitable[Env[ObservationT, ActionT]],
    ],
    policy: Policy[ObservationT, ActionT] | None = None,
    *,
    batch_policy: BatchPolicy[ObservationT, ActionT] | None = None,
    num_rollouts: int = 1,
    item_id: str | None = None,
    max_steps: int | None = None,
    concurrency: int | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> TrajectoryGroup[ItemT, ObservationT, ActionT]:
    """Collect an async trajectory group with a scalar or active batch policy.

    Supply exactly one of ``policy`` and ``batch_policy``. Scalar policies use
    the established independent-episode collector. Batch policies receive only
    unfinished observations on each wave, so uneven episode lengths require no
    caller-side handling.

    Args:
        item: Input item used to create every environment in the group.
        make_env: Sync or async factory returning fresh async environments.
        policy: Optional scalar async-compatible policy.
        batch_policy: Optional async-compatible active-observation policy.
        num_rollouts: Number of trajectories to collect.
        item_id: Optional stable identifier for the item.
        max_steps: Optional per-trajectory action limit.
        concurrency: Optional environment creation, reset, and step limit.
            Defaults to one for scalar collection and all group members for
            batch collection.
        metadata: Optional metadata attached to the trajectory group.

    Returns:
        A trajectory group in environment-creation order.

    Raises:
        ValueError: If exactly one policy is not supplied or an argument is
            invalid.
    """
    if (policy is None) == (batch_policy is None):
        raise ValueError("provide exactly one of policy or batch_policy")
    if concurrency is not None and concurrency < 1:
        raise ValueError("concurrency must be at least 1")
    if batch_policy is not None:
        return await _arollout_group_batch(
            item,
            make_env,
            batch_policy,
            num_rollouts=num_rollouts,
            item_id=item_id,
            max_steps=max_steps,
            concurrency=concurrency,
            metadata=metadata,
        )
    return await _arollout_group_scalar(
        item,
        make_env,
        cast(Policy[ObservationT, ActionT], policy),
        num_rollouts=num_rollouts,
        item_id=item_id,
        max_steps=max_steps,
        concurrency=concurrency or 1,
        metadata=metadata,
    )


__all__ = [
    "ActionT",
    "BatchPolicy",
    "Policy",
    "BatchPolicy",
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
