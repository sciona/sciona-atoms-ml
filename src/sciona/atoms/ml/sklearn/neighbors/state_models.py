"""State containers for sklearn neighbors atoms."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class NeighborsGraphTransformerState:
    """Fitted dense neighbor-graph transformer state."""

    training_data: NDArray[np.float64]
    mode: str
    n_neighbors: int | None
    radius: float | None
    metric: str
    p: float
    transformer_kind: str
    n_features_in: int


@dataclass(frozen=True)
class NeighborsRegressorState:
    """Fitted dense neighbor-regression state."""

    training_data: NDArray[np.float64]
    target: NDArray[np.float64]
    weights: str
    n_neighbors: int | None
    radius: float | None
    metric: str
    p: float
    regressor_kind: str
    n_features_in: int
    outputs_2d: bool


@dataclass(frozen=True)
class NeighborsClassifierState:
    """Fitted dense neighbor-classification state."""

    training_data: NDArray[np.float64]
    labels: NDArray[np.int64]
    classes: NDArray[np.float64]
    weights: str
    n_neighbors: int | None
    radius: float | None
    metric: str
    p: float
    classifier_kind: str
    n_features_in: int


@dataclass(frozen=True)
class NearestNeighborsState:
    """Fitted dense nearest-neighbor search state."""

    training_data: NDArray[np.float64]
    n_neighbors: int
    radius: float
    metric: str
    p: float
    n_features_in: int


@dataclass(frozen=True)
class NearestCentroidState:
    """Fitted dense nearest-centroid classifier state."""

    classes: NDArray[np.float64]
    centroids: NDArray[np.float64]
    deviations: NDArray[np.float64]
    within_class_std_dev: NDArray[np.float64]
    class_prior: NDArray[np.float64]
    metric: str
    shrink_threshold: float | None
    n_features_in: int
