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
class IncrementalPCAState:
    """Fitted incremental PCA components and running feature statistics."""

    components: NDArray[np.float64]
    explained_variance: NDArray[np.float64]
    explained_variance_ratio: NDArray[np.float64]
    singular_values: NDArray[np.float64]
    mean: NDArray[np.float64]
    var: NDArray[np.float64]
    noise_variance: float
    n_samples_seen: int
    n_components: int
    n_features_in: int
    whiten: bool
    batch_size: int | None


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


@dataclass(frozen=True)
class KernelPCAState:
    """Fitted dense linear-kernel PCA eigensystem and centering statistics."""

    eigenvalues: NDArray[np.float64]
    eigenvectors: NDArray[np.float64]
    X_fit: NDArray[np.float64]
    kernel_centerer_rows: NDArray[np.float64]
    kernel_centerer_all: float
    n_components: int
    n_features_in: int
    gamma: float
    kernel: str
    eigen_solver: str
    remove_zero_eig: bool
    fit_inverse_transform: bool


@dataclass(frozen=True)
class FactorAnalysisState:
    """Fitted factor loading matrix and diagonal noise variances."""

    components: NDArray[np.float64]
    noise_variance: NDArray[np.float64]
    mean: NDArray[np.float64]
    loglike: NDArray[np.float64]
    n_iter: int
    n_components: int
    n_features_in: int
    tol: float
    max_iter: int
    svd_method: str
    rotation: None
