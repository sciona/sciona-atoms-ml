"""State containers for sklearn covariance estimator atoms."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class CovarianceState:
    """Fitted covariance estimate, optional precision, and estimator metadata."""

    covariance: NDArray[np.float64]
    location: NDArray[np.float64]
    precision: NDArray[np.float64] | None
    store_precision: bool
    assume_centered: bool
    estimator: str
    shrinkage: float | None
    n_features_in: int
