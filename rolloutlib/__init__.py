"""Gymnasium-style environments and rollout primitives for post-training."""

from importlib.metadata import version

from . import datasets, envs, evals, graders, rollouts, spaces
from .datasets import Dataset, RLDataset
from .envs import (
    AsyncEnv,
    AsyncGradingWrapper,
    AsyncSingleTurnEnv,
    Env,
    GradingWrapper,
    SingleTurnEnv,
    as_async,
    as_sync,
    check_async_env,
)
from .graders import Score
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

__version__ = version("rolloutlib")

__all__ = [
    "AsyncEnv",
    "AsyncGradingWrapper",
    "AsyncPolicy",
    "AsyncSingleTurnEnv",
    "Dataset",
    "Env",
    "GradingWrapper",
    "Policy",
    "PolicyOutput",
    "RLDataset",
    "RolloutError",
    "Score",
    "SingleTurnEnv",
    "Step",
    "Trajectory",
    "TrajectoryGroup",
    "__version__",
    "as_async",
    "as_sync",
    "arollout",
    "arollout_group",
    "check_async_env",
    "datasets",
    "envs",
    "evals",
    "graders",
    "rollout",
    "rollout_group",
    "rollouts",
    "spaces",
]
