"""Ghost witnesses for shared covariance fit-bookkeeping atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_covariance_fit_location(
    X: AbstractArray,
    *,
    assume_centered: bool,
) -> AbstractArray:
    """Describe the fitted location vector resolved during covariance fit."""
    del assume_centered
    if len(X.shape) != 2:
        raise ValueError("X must be 2D")
    return AbstractArray(shape=(int(X.shape[1]),), dtype="float64")


def witness_covariance_set_covariance_matrix(
    covariance: AbstractArray,
) -> AbstractArray:
    """Describe the validated covariance_ matrix stored by `_set_covariance`."""
    if len(covariance.shape) != 2:
        raise ValueError("covariance must be 2D")
    if int(covariance.shape[0]) != int(covariance.shape[1]):
        raise ValueError("covariance must be square")
    return AbstractArray(shape=covariance.shape, dtype="float64")


def witness_covariance_set_precision_required(
    store_precision: bool,
) -> bool:
    """Describe the branch predicate for storing precision_."""
    del store_precision
    return False


def witness_covariance_set_precision_matrix(
    covariance: AbstractArray,
) -> AbstractArray:
    """Describe the precision matrix derived from a stored covariance matrix."""
    if len(covariance.shape) != 2:
        raise ValueError("covariance must be 2D")
    if int(covariance.shape[0]) != int(covariance.shape[1]):
        raise ValueError("covariance must be square")
    return AbstractArray(shape=covariance.shape, dtype="float64")
