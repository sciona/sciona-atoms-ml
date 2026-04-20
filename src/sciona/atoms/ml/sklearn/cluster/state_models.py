"""State containers for sklearn cluster atoms."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class AffinityPropagationState:
    """Fitted affinity propagation state and metadata."""

    cluster_centers_indices: NDArray[np.int_]
    labels: NDArray[np.int_]
    n_iter: int
    affinity_matrix: NDArray[np.float64]
    cluster_centers: NDArray[np.float64] | None
    affinity: str
    preference: float | NDArray[np.float64]
    damping: float
    n_features_in: int
