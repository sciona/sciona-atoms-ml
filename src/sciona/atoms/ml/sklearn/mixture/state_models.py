"""State containers for sklearn mixture atoms."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class GaussianMixtureDiagState:
    """Fitted diagonal-covariance Gaussian mixture state."""

    weights: NDArray[np.float64]
    means: NDArray[np.float64]
    covariances: NDArray[np.float64]
    precisions_cholesky: NDArray[np.float64]
    converged: bool
    n_iter: int
    lower_bound: float
    lower_bounds: NDArray[np.float64]
    reg_covar: float
    n_features_in: int
