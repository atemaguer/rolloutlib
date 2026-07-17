"""Gymnasium-style environments and rollout primitives for post-training."""

from importlib.metadata import version

from . import content, datasets, envs, evals, graders, policies, rollouts, spaces, wrappers
from .datasets import Dataset, RLDataset
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
    AsyncPolicy,
    Policy,
    PolicyOutput,
    RolloutError,
    Step,
    Trajectory,
    TrajectoryGroup,
    arollout,
    arollout_group,
    rollout,
    rollout_group,
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
    "AsyncGradingWrapper",
    "AsyncGrader",
    "AsyncPolicy",
    "AsyncSingleTurnEnv",
    "ChatObservationWrapper",
    "ChatHistoryWrapper",
    "Dataset",
    "Criterion",
    "Env",
    "GradingWrapper",
    "Grader",
    "Level",
    "Policy",
    "PolicyOutput",
    "RLDataset",
    "RolloutError",
    "Rubric",
    "Score",
    "SingleTurnEnv",
    "Step",
    "Trajectory",
    "TrajectoryGroup",
    "ToolCallActionWrapper",
    "__version__",
    "as_async",
    "as_sync",
    "arollout",
    "arollout_group",
    "check_async_env",
    "content",
    "datasets",
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
]
