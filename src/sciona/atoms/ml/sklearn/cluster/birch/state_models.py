"""State containers for BIRCH no-global-clustering atoms."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class BirchNoGlobalState:
    """Immutable BIRCH CF-tree summary for n_clusters=None."""

    subcluster_centers: NDArray[np.float64]
    subcluster_labels: NDArray[np.int_]
    labels: NDArray[np.int_] | None
    threshold: float
    branching_factor: int
    compute_labels: bool
    n_features_in: int
    n_features_out: int
