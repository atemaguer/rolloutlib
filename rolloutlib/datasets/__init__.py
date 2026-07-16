"""Small data-source containers used by rolloutlib.

Datasets are sources of items to process. They are deliberately independent
from trajectories: rollout generation consumes a dataset and produces
trajectories or trajectory groups.
"""

from .core import Dataset, RLDataset

__all__ = ["Dataset", "RLDataset"]
