"""State containers for sklearn decomposition atoms."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class PCAState:
    """Fitted PCA components and variance metadata."""

    components: NDArray[np.float64]
    explained_variance: NDArray[np.float64]
    explained_variance_ratio: NDArray[np.float64]
    singular_values: NDArray[np.float64]
    mean: NDArray[np.float64]
    noise_variance: float
    n_samples: int
    n_components: int
    n_features_in: int
    whiten: bool
    svd_solver: str


@dataclass(frozen=True)
class TruncatedSVDState:
    """Fitted truncated SVD components and variance metadata."""

    components: NDArray[np.float64]
    explained_variance: NDArray[np.float64]
    explained_variance_ratio: NDArray[np.float64]
    singular_values: NDArray[np.float64]
    n_components: int
    n_features_in: int
    algorithm: str
    n_iter: int
    n_oversamples: int
    power_iteration_normalizer: str
    random_state: int | None
    tol: float
