"""State containers for sklearn VarianceThreshold atoms."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class VarianceThresholdState:
    """Learned variances and threshold for a fitted variance selector."""

    variances: NDArray[np.float64]
    threshold: float
    n_features_in: int
