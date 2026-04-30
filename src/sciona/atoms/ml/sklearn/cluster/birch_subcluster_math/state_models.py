"""State containers for BIRCH subcluster math helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class BirchSubclusterStats:
    """Immutable summary of one sklearn BIRCH _CFSubcluster."""

    n_samples: int
    linear_sum: NDArray[np.float64]
    squared_sum: float
    centroid: NDArray[np.float64]
    sq_norm: float
