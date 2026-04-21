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
