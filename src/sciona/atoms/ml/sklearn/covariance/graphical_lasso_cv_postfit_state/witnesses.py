"""Ghost witnesses for GraphicalLassoCV postfit-state atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_graphical_lasso_cv_fit_alpha(
    alpha: float,
) -> AbstractArray:
    """Describe GraphicalLassoCV.alpha_ postfit exposure."""
    del alpha
    return AbstractArray(shape=(), dtype="float64")


def witness_graphical_lasso_cv_fit_covariance(
    covariance: AbstractArray,
) -> AbstractArray:
    """Describe GraphicalLassoCV.covariance_ postfit exposure."""
    return AbstractArray(shape=covariance.shape, dtype=covariance.dtype)


def witness_graphical_lasso_cv_fit_precision(
    precision: AbstractArray,
) -> AbstractArray:
    """Describe GraphicalLassoCV.precision_ postfit exposure."""
    return AbstractArray(shape=precision.shape, dtype=precision.dtype)


def witness_graphical_lasso_cv_fit_costs(
    costs: AbstractArray,
) -> AbstractArray:
    """Describe GraphicalLassoCV.costs_ postfit exposure."""
    return AbstractArray(shape=costs.shape, dtype=costs.dtype)


def witness_graphical_lasso_cv_fit_n_iter(
    n_iter: int,
) -> AbstractArray:
    """Describe GraphicalLassoCV.n_iter_ postfit exposure."""
    del n_iter
    return AbstractArray(shape=(), dtype="int64")


def witness_graphical_lasso_cv_fit_return_self(
    estimator_token: str,
) -> AbstractArray:
    """Describe GraphicalLassoCV.fit returning self."""
    del estimator_token
    return AbstractArray(shape=(), dtype="str")
