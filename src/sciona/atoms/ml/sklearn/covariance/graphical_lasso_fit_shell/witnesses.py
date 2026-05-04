"""Ghost witnesses for GraphicalLasso fit-shell atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_graphical_lasso_fit_use_precomputed_covariance(
    covariance_mode: str,
) -> bool:
    """Describe GraphicalLasso's precomputed-covariance branch predicate."""
    del covariance_mode
    return False


def witness_graphical_lasso_fit_empirical_covariance(
    X: AbstractArray,
    covariance_mode: str,
    *,
    assume_centered: bool,
) -> AbstractArray:
    """Describe the covariance matrix passed into GraphicalLasso's deferred solver."""
    del covariance_mode, assume_centered
    return AbstractArray(shape=(int(X.shape[1]), int(X.shape[1])), dtype="float64")


def witness_graphical_lasso_fit_location(
    X: AbstractArray,
    covariance_mode: str,
    *,
    assume_centered: bool,
) -> AbstractArray:
    """Describe GraphicalLasso.location_ before the deferred solver call."""
    del covariance_mode, assume_centered
    return AbstractArray(shape=(int(X.shape[1]),), dtype="float64")


def witness_graphical_lasso_fit_covariance(
    covariance: AbstractArray,
) -> AbstractArray:
    """Describe GraphicalLasso.covariance_ postfit exposure."""
    return AbstractArray(shape=covariance.shape, dtype=covariance.dtype)


def witness_graphical_lasso_fit_precision(
    precision: AbstractArray,
) -> AbstractArray:
    """Describe GraphicalLasso.precision_ postfit exposure."""
    return AbstractArray(shape=precision.shape, dtype=precision.dtype)


def witness_graphical_lasso_fit_costs(
    costs: AbstractArray,
) -> AbstractArray:
    """Describe GraphicalLasso.costs_ postfit exposure."""
    return AbstractArray(shape=costs.shape, dtype=costs.dtype)


def witness_graphical_lasso_fit_n_iter(
    n_iter: int,
) -> AbstractArray:
    """Describe GraphicalLasso.n_iter_ postfit exposure."""
    del n_iter
    return AbstractArray(shape=(), dtype="int64")


def witness_graphical_lasso_fit_return_self(
    estimator_token: str,
) -> AbstractArray:
    """Describe GraphicalLasso.fit returning self."""
    del estimator_token
    return AbstractArray(shape=(), dtype="str")
