"""Ghost witnesses for GaussianProcessRegressor postfit-attribute atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_gp_regression_fit_L(
    L: AbstractArray,
) -> AbstractArray:
    """Describe GaussianProcessRegressor.L_ postfit exposure."""
    return AbstractArray(shape=L.shape, dtype=L.dtype)


def witness_gp_regression_fit_alpha(
    alpha: AbstractArray,
) -> AbstractArray:
    """Describe GaussianProcessRegressor.alpha_ postfit exposure."""
    return AbstractArray(shape=alpha.shape, dtype=alpha.dtype)


def witness_gp_regression_fit_log_marginal_likelihood_value(
    value: float,
) -> AbstractArray:
    """Describe GaussianProcessRegressor.log_marginal_likelihood_value_ exposure."""
    del value
    return AbstractArray(shape=(), dtype="float64")


def witness_gp_regression_fit_return_self(
    estimator_token: str,
) -> AbstractArray:
    """Describe GaussianProcessRegressor.fit returning self."""
    del estimator_token
    return AbstractArray(shape=(), dtype="str")
