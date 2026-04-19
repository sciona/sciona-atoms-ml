"""State containers for sklearn semi-supervised atoms."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray


MatrixLike = NDArray[np.float64] | sp.spmatrix
GraphKernel = str | Callable[[MatrixLike, MatrixLike], MatrixLike]


@dataclass(frozen=True)
class LabelPropagationState:
    """Fitted graph-label distributions for propagation-style classifiers."""

    X: MatrixLike
    classes: NDArray[np.object_]
    label_distributions: NDArray[np.float64]
    transduction: NDArray[np.object_]
    n_iter: int
    kernel: GraphKernel
    gamma: float
    n_neighbors: int
    alpha: float | None
    variant: str
    n_jobs: int | None
    n_features_in: int


@dataclass(frozen=True)
class SelfTrainingClassifierState:
    """Fitted self-training classifier state and pseudo-label history."""

    estimator: object
    classes: NDArray[np.object_]
    transduction: NDArray[np.object_]
    labeled_iter: NDArray[np.int_]
    n_iter: int
    termination_condition: str
    threshold: float
    criterion: str
    k_best: int
    max_iter: int | None
    n_features_in: int
