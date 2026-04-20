"""State containers for sklearn linear model atoms."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class LinearRegressionState:
    """Fitted coefficients and intercept for ordinary least squares."""

    coef: NDArray[np.float64]
    intercept: NDArray[np.float64]
    rank: int
    singular: NDArray[np.float64]
    fit_intercept: bool
    n_features_in: int
    n_outputs: int


@dataclass(frozen=True)
class RidgeState:
    """Fitted coefficients and intercept for dense ridge regression."""

    coef: NDArray[np.float64]
    intercept: NDArray[np.float64]
    alpha: NDArray[np.float64]
    fit_intercept: bool
    solver: str
    n_features_in: int
    n_outputs: int


@dataclass(frozen=True)
class RidgeCVState:
    """Fitted dense ridge regression state with selected CV alpha."""

    coef: NDArray[np.float64]
    intercept: NDArray[np.float64]
    alpha: NDArray[np.float64]
    best_score: NDArray[np.float64]
    fit_intercept: bool
    n_features_in: int
    n_outputs: int


@dataclass(frozen=True)
class RidgeClassifierState:
    """Fitted coefficients and classes for dense ridge classification."""

    coef: NDArray[np.float64]
    intercept: NDArray[np.float64]
    classes: NDArray[np.float64]
    alpha: NDArray[np.float64]
    fit_intercept: bool
    solver: str
    n_features_in: int
