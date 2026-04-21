"""State containers for sklearn manifold atoms."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ClassicalMDSState:
    """Fitted classical MDS coordinates and source dissimilarities."""

    embedding: NDArray[np.float64]
    dissimilarity_matrix: NDArray[np.float64]
    eigenvalues: NDArray[np.float64]
    n_components: int
    metric: str
    n_features_in: int


@dataclass(frozen=True)
class SMACOFState:
    """Metric SMACOF embedding and final stress."""

    embedding: NDArray[np.float64]
    stress: float
    n_iter: int
    dissimilarity_matrix: NDArray[np.float64]
    n_components: int
    normalized_stress: bool


@dataclass(frozen=True)
class MDSState:
    """Fitted metric MDS embedding state."""

    embedding: NDArray[np.float64]
    stress: float
    n_iter: int
    dissimilarity_matrix: NDArray[np.float64]
    n_components: int
    metric: str
    metric_mds: bool
    normalized_stress: bool
    n_features_in: int


@dataclass(frozen=True)
class SpectralEmbeddingState:
    """Fitted dense spectral embedding state."""

    embedding: NDArray[np.float64]
    affinity_matrix: NDArray[np.float64]
    n_components: int
    affinity: str
    gamma: float | None
    eigen_solver: str
    n_features_in: int


@dataclass(frozen=True)
class IsomapState:
    """Fitted dense Isomap embedding and geodesic-kernel state."""

    embedding: NDArray[np.float64]
    dist_matrix: NDArray[np.float64]
    training_data: NDArray[np.float64]
    eigenvalues: NDArray[np.float64]
    eigenvectors: NDArray[np.float64]
    kernel_centerer_rows: NDArray[np.float64]
    kernel_centerer_all: float
    n_neighbors: int
    n_components: int
    path_method: str
    metric: str
    p: float
    n_features_in: int


@dataclass(frozen=True)
class LocallyLinearEmbeddingState:
    """Fitted dense standard LLE embedding and reconstruction state."""

    embedding: NDArray[np.float64]
    reconstruction_error: float
    training_data: NDArray[np.float64]
    weights: NDArray[np.float64]
    reconstruction_matrix: NDArray[np.float64]
    n_neighbors: int
    n_components: int
    reg: float
    eigen_solver: str
    method: str
    n_features_in: int
