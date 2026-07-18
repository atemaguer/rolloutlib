"""Gymnasium-native infrastructure for agentic RL post-training."""

from importlib.metadata import version

from . import content, envs, evals, graders, policies, rollouts, spaces, wrappers
from .envs import (
    AsyncEnv,
    AsyncSingleTurnEnv,
    Env,
    SingleTurnEnv,
    as_async,
    as_sync,
    check_async_env,
)
from .graders import AsyncGrader, Criterion, Grader, Level, Rubric, Score
from .rollouts import (
    AsyncBatchPolicy,
    AsyncPolicy,
    BatchPolicy,
    Policy,
    PolicyOutput,
    RolloutError,
    Step,
    Trajectory,
    TrajectoryGroup,
    abatched_rollout_group,
    arollout,
    arollout_group,
    rollout,
    rollout_group,
    vector_rollout_group,
)
from .wrappers import (
    AsyncGradingWrapper,
    ChatHistoryWrapper,
    ChatObservationWrapper,
    GradingWrapper,
    ToolCallActionWrapper,
    wrap_language_env,
)

__version__ = version("rolloutlib")

__all__ = [
    "AsyncEnv",
    "AsyncBatchPolicy",
    "AsyncGradingWrapper",
    "AsyncGrader",
    "AsyncPolicy",
    "AsyncSingleTurnEnv",
    "BatchPolicy",
    "ChatObservationWrapper",
    "ChatHistoryWrapper",
    "Criterion",
    "Env",
    "GradingWrapper",
    "Grader",
    "Level",
    "Policy",
    "PolicyOutput",
    "RolloutError",
    "Rubric",
    "Score",
    "SingleTurnEnv",
    "Step",
    "Trajectory",
    "TrajectoryGroup",
    "ToolCallActionWrapper",
    "__version__",
    "abatched_rollout_group",
    "as_async",
    "as_sync",
    "arollout",
    "arollout_group",
    "check_async_env",
    "content",
    "envs",
    "evals",
    "graders",
    "policies",
    "rollout",
    "rollout_group",
    "rollouts",
    "spaces",
    "wrappers",
    "wrap_language_env",
    "vector_rollout_group",
]
