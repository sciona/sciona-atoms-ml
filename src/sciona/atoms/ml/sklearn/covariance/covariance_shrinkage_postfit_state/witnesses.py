"""Ghost witnesses for covariance shrinkage post-fit state atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_covariance_ledoit_wolf_fit_shrinkage(
    shrinkage: float,
) -> AbstractArray:
    """Describe the fitted LedoitWolf shrinkage scalar."""
    del shrinkage
    return AbstractArray(shape=(), dtype="float64")


def witness_covariance_oas_fit_shrinkage(
    shrinkage: float,
) -> AbstractArray:
    """Describe the fitted OAS shrinkage scalar."""
    del shrinkage
    return AbstractArray(shape=(), dtype="float64")


def witness_covariance_fit_return_self(
    estimator_token: str,
) -> AbstractArray:
    """Describe covariance-estimator fit returning self."""
    del estimator_token
    return AbstractArray(shape=(), dtype="str")
