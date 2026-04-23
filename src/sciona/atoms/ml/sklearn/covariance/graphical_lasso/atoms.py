"""Graphical-lasso scoring atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_graphical_lasso_dual_gap,
    witness_graphical_lasso_log_likelihood,
    witness_graphical_lasso_objective,
    witness_graphical_lasso_offdiag_l1_penalty,
)


def _finite_square_matrix(values: NDArray[np.float64]) -> bool:
    try:
        matrix = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(matrix.ndim == 2 and matrix.shape[0] >= 1 and matrix.shape[0] == matrix.shape[1] and np.all(np.isfinite(matrix)))


def _finite_symmetric_square(values: NDArray[np.float64]) -> bool:
    if not _finite_square_matrix(values):
        return False
    matrix = np.asarray(values, dtype=np.float64)
    return bool(np.allclose(matrix, matrix.T))


def _precision_valid(precision: NDArray[np.float64]) -> bool:
    if not _finite_symmetric_square(precision):
        return False
    sign, logdet = np.linalg.slogdet(np.asarray(precision, dtype=np.float64))
    return bool(sign > 0 and np.isfinite(logdet))


def _same_square_shape(emp_cov: NDArray[np.float64], precision: NDArray[np.float64]) -> bool:
    return bool(_finite_symmetric_square(emp_cov) and _precision_valid(precision) and np.asarray(emp_cov).shape == np.asarray(precision).shape)


def _same_square_shape_for_gap(emp_cov: NDArray[np.float64], precision: NDArray[np.float64]) -> bool:
    return bool(_finite_symmetric_square(emp_cov) and _finite_symmetric_square(precision) and np.asarray(emp_cov).shape == np.asarray(precision).shape)


def _nonnegative_finite(value: float) -> bool:
    return bool(isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(float(value)) and float(value) >= 0.0)


@register_atom(witness_graphical_lasso_offdiag_l1_penalty)
@icontract.require(lambda precision: _finite_square_matrix(precision), "precision must be a finite square matrix")
@icontract.ensure(lambda result: np.isfinite(result) and result >= 0.0, "off-diagonal penalty must be finite and nonnegative")
def graphical_lasso_offdiag_l1_penalty(precision: NDArray[np.float64]) -> float:
    """Compute the unpenalized-diagonal L1 term used by graphical lasso."""
    values = np.asarray(precision, dtype=np.float64)
    return float(np.abs(values).sum() - np.abs(np.diag(values)).sum())


@register_atom(witness_graphical_lasso_log_likelihood)
@icontract.require(lambda emp_cov, precision: _same_square_shape(emp_cov, precision), "emp_cov and precision must be compatible finite symmetric matrices")
@icontract.ensure(lambda result: np.isfinite(result), "log-likelihood must be finite")
def graphical_lasso_log_likelihood(
    emp_cov: NDArray[np.float64],
    precision: NDArray[np.float64],
) -> float:
    """Compute sklearn's Gaussian covariance log-likelihood score."""
    emp_values = np.asarray(emp_cov, dtype=np.float64)
    precision_values = np.asarray(precision, dtype=np.float64)
    logdet = float(np.linalg.slogdet(precision_values)[1])
    n_features = precision_values.shape[0]
    score = -float(np.sum(emp_values * precision_values)) + logdet
    score -= n_features * np.log(2.0 * np.pi)
    score /= 2.0
    return float(score)


@register_atom(witness_graphical_lasso_objective)
@icontract.require(lambda mle, precision, alpha: _same_square_shape(mle, precision) and _nonnegative_finite(alpha), "mle, precision, and alpha must be compatible")
@icontract.ensure(lambda result: np.isfinite(result), "objective value must be finite")
def graphical_lasso_objective(
    mle: NDArray[np.float64],
    precision: NDArray[np.float64],
    alpha: float,
) -> float:
    """Evaluate sklearn's graphical-lasso objective at a precision matrix."""
    n_features = np.asarray(precision, dtype=np.float64).shape[0]
    cost = -2.0 * graphical_lasso_log_likelihood(mle, precision)
    cost += n_features * np.log(2.0 * np.pi)
    cost += float(alpha) * graphical_lasso_offdiag_l1_penalty(precision)
    return float(cost)


@register_atom(witness_graphical_lasso_dual_gap)
@icontract.require(lambda emp_cov, precision, alpha: _same_square_shape_for_gap(emp_cov, precision) and _nonnegative_finite(alpha), "emp_cov, precision, and alpha must be compatible")
@icontract.ensure(lambda result: np.isfinite(result), "dual-gap value must be finite")
def graphical_lasso_dual_gap(
    emp_cov: NDArray[np.float64],
    precision: NDArray[np.float64],
    alpha: float,
) -> float:
    """Compute sklearn's graphical-lasso dual-gap convergence score."""
    covariance_values = np.asarray(emp_cov, dtype=np.float64)
    precision_values = np.asarray(precision, dtype=np.float64)
    gap = float(np.sum(covariance_values * precision_values))
    gap -= precision_values.shape[0]
    gap += float(alpha) * graphical_lasso_offdiag_l1_penalty(precision_values)
    return float(gap)
