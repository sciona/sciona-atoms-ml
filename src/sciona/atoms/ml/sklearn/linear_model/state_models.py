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


@dataclass(frozen=True)
class RidgeClassifierCVState:
    """Fitted dense ridge classifier state with selected CV alpha."""

    coef: NDArray[np.float64]
    intercept: NDArray[np.float64]
    classes: NDArray[np.float64]
    alpha: NDArray[np.float64]
    best_score: NDArray[np.float64]
    fit_intercept: bool
    n_features_in: int


@dataclass(frozen=True)
class OrthogonalMatchingPursuitState:
    """Fitted dense orthogonal matching pursuit coefficients."""

    coef: NDArray[np.float64]
    intercept: NDArray[np.float64]
    n_iter: NDArray[np.int64]
    n_nonzero_coefs: int | None
    tol: float | None
    fit_intercept: bool
    precompute: bool | str
    n_features_in: int
    n_outputs: int


@dataclass(frozen=True)
class OrthogonalMatchingPursuitCVState:
    """Fitted dense OMP state with a cross-validated sparsity level."""

    coef: NDArray[np.float64]
    intercept: NDArray[np.float64]
    n_iter: NDArray[np.int64]
    n_nonzero_coefs: int
    max_iter: int
    cv: int
    fit_intercept: bool
    n_features_in: int


@dataclass(frozen=True)
class BayesianRidgeState:
    """Fitted Bayesian ridge posterior mean and covariance state."""

    coef: NDArray[np.float64]
    intercept: float
    alpha: float
    lambda_: float
    sigma: NDArray[np.float64]
    scores: NDArray[np.float64]
    n_iter: int
    x_offset: NDArray[np.float64]
    x_scale: NDArray[np.float64]
    fit_intercept: bool
    compute_score: bool
    n_features_in: int


@dataclass(frozen=True)
class ARDRegressionState:
    """Fitted automatic relevance determination regression state."""

    coef: NDArray[np.float64]
    intercept: float
    alpha: float
    lambda_: NDArray[np.float64]
    sigma: NDArray[np.float64]
    scores: NDArray[np.float64]
    n_iter: int
    threshold_lambda: float
    x_offset: NDArray[np.float64]
    x_scale: NDArray[np.float64]
    fit_intercept: bool
    compute_score: bool
    n_features_in: int


@dataclass(frozen=True)
class TheilSenRegressorState:
    """Fitted dense Theil-Sen regression state."""

    coef: NDArray[np.float64]
    intercept: float
    breakdown: float
    n_iter: int
    n_subpopulation: int
    n_subsamples: int
    fit_intercept: bool
    max_subpopulation: int
    n_features_in: int


@dataclass(frozen=True)
class LarsPathState:
    """Dense least-angle-regression path state."""

    alphas: NDArray[np.float64]
    active: NDArray[np.int64]
    coefs: NDArray[np.float64]
    n_iter: int
    method: str
    alpha_min: float
    n_samples: int
    n_features_in: int


@dataclass(frozen=True)
class LarsState:
    """Fitted dense least-angle-regression state."""

    coef: NDArray[np.float64]
    intercept: float
    alphas: NDArray[np.float64]
    active: NDArray[np.int64]
    coef_path: NDArray[np.float64]
    n_iter: int
    fit_intercept: bool
    n_nonzero_coefs: int
    n_features_in: int


@dataclass(frozen=True)
class LassoLarsState:
    """Fitted dense Lasso-LARS regression state."""

    coef: NDArray[np.float64]
    intercept: float
    alpha: float
    alphas: NDArray[np.float64]
    active: NDArray[np.int64]
    coef_path: NDArray[np.float64]
    n_iter: int
    fit_intercept: bool
    max_iter: int
    n_features_in: int


@dataclass(frozen=True)
class LassoLarsICState:
    """Fitted dense Lasso-LARS information-criterion state."""

    coef: NDArray[np.float64]
    intercept: float
    alpha: float
    alphas: NDArray[np.float64]
    criterion_values: NDArray[np.float64]
    noise_variance: float
    n_iter: int
    criterion: str
    fit_intercept: bool
    max_iter: int
    n_features_in: int
