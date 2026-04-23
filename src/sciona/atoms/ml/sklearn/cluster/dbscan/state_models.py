"""State containers for DBSCAN boundary atoms."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class DBSCANState:
    """Immutable DBSCAN fit output and fitted metadata."""

    core_sample_indices: NDArray[np.int_]
    labels: NDArray[np.int_]
    components: NDArray[np.float64]
    eps: float
    min_samples: int
    metric: str
    algorithm: str
    leaf_size: int
    p: float | None
    n_features_in: int
