"""Ghost witnesses for graphical-lasso scoring atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def _check_square(matrix: AbstractArray, name: str) -> int:
    if len(matrix.shape) != 2:
        raise ValueError(f"{name} must be 2D")
    if int(matrix.shape[0]) < 1 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be nonempty and square")
    return int(matrix.shape[0])


def _check_pair(emp_cov: AbstractArray, precision: AbstractArray) -> int:
    n_features = _check_square(emp_cov, "emp_cov")
    if _check_square(precision, "precision") != n_features:
        raise ValueError("matrix shapes must match")
    return n_features


def _check_alpha(alpha: float) -> None:
    if alpha < 0.0:
        raise ValueError("alpha must be nonnegative")


def witness_graphical_lasso_offdiag_l1_penalty(precision: AbstractArray) -> float:
    """Describe a nonnegative off-diagonal L1 penalty."""
    _check_square(precision, "precision")
    return 0.0


def witness_graphical_lasso_log_likelihood(
    emp_cov: AbstractArray,
    precision: AbstractArray,
) -> float:
    """Describe Gaussian covariance log-likelihood scoring."""
    _check_pair(emp_cov, precision)
    return 0.0


def witness_graphical_lasso_objective(
    mle: AbstractArray,
    precision: AbstractArray,
    alpha: float,
) -> float:
    """Describe graphical-lasso objective scoring."""
    _check_pair(mle, precision)
    _check_alpha(alpha)
    return 0.0


def witness_graphical_lasso_dual_gap(
    emp_cov: AbstractArray,
    precision: AbstractArray,
    alpha: float,
) -> float:
    """Describe graphical-lasso dual-gap scoring."""
    _check_pair(emp_cov, precision)
    _check_alpha(alpha)
    return 0.0
