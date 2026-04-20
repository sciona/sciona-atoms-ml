"""State containers for sklearn discriminant-analysis atoms."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class QDAState:
    """Fitted Quadratic Discriminant Analysis means and covariance factors."""

    classes: NDArray[np.float64]
    priors: NDArray[np.float64]
    means: NDArray[np.float64]
    scalings: tuple[NDArray[np.float64], ...]
    rotations: tuple[NDArray[np.float64], ...]
    covariance: tuple[NDArray[np.float64], ...] | None
    reg_param: float
    store_covariance: bool
    n_features_in: int


@dataclass(frozen=True)
class LDAState:
    """Fitted linear discriminant means, weights, and projection basis."""

    classes: NDArray[np.float64]
    priors: NDArray[np.float64]
    means: NDArray[np.float64]
    xbar: NDArray[np.float64]
    scalings: NDArray[np.float64]
    coef: NDArray[np.float64]
    intercept: NDArray[np.float64]
    explained_variance_ratio: NDArray[np.float64]
    covariance: NDArray[np.float64] | None
    n_components: int
    n_features_in: int
    store_covariance: bool
    tol: float
