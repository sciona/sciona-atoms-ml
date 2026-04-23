"""State containers for limited HDBSCAN boundary atoms."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class HDBSCANState:
    """Immutable HDBSCAN fit output and fitted metadata."""

    labels: NDArray[np.int_]
    probabilities: NDArray[np.float64]
    single_linkage_tree: NDArray[np.generic]
    min_cluster_size: int
    min_samples: int
    cluster_selection_epsilon: float
    max_cluster_size: int | None
    metric: str
    alpha: float
    algorithm: str
    leaf_size: int
    cluster_selection_method: str
    allow_single_cluster: bool
    n_features_in: int
