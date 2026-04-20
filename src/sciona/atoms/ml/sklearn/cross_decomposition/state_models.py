"""State containers for sklearn cross-decomposition atoms."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class PLSSVDState:
    """Fitted PLS-SVD weights and centering/scaling metadata."""

    x_weights: NDArray[np.float64]
    y_weights: NDArray[np.float64]
    singular_values: NDArray[np.float64]
    x_mean: NDArray[np.float64]
    y_mean: NDArray[np.float64]
    x_std: NDArray[np.float64]
    y_std: NDArray[np.float64]
    n_components: int
    scale: bool
    n_features_in: int
    n_targets: int


@dataclass(frozen=True)
class PLSState:
    """Fitted PLS/CCA weights, scores, loadings, rotations, and metadata."""

    x_weights: NDArray[np.float64]
    y_weights: NDArray[np.float64]
    x_scores: NDArray[np.float64]
    y_scores: NDArray[np.float64]
    x_loadings: NDArray[np.float64]
    y_loadings: NDArray[np.float64]
    x_rotations: NDArray[np.float64]
    y_rotations: NDArray[np.float64]
    coef: NDArray[np.float64]
    intercept: NDArray[np.float64]
    x_mean: NDArray[np.float64]
    y_mean: NDArray[np.float64]
    x_std: NDArray[np.float64]
    y_std: NDArray[np.float64]
    n_iter: tuple[int, ...]
    n_components: int
    scale: bool
    deflation_mode: str
    mode: str
    algorithm: str
    n_features_in: int
    n_targets: int
    predict_1d: bool
