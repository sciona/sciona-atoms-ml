"""Ghost witnesses for shared covariance post-fit API helpers."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_covariance_precision_matrix(
    state: object,
) -> AbstractArray:
    """Describe the precision matrix resolved from a fitted covariance state."""
    del state
    return AbstractArray(shape=("n_features", "n_features"), dtype="float64")


def witness_covariance_score_test_covariance(
    X_test: AbstractArray,
    location: AbstractArray,
) -> AbstractArray:
    """Describe the centered empirical covariance used by covariance score methods."""
    if len(X_test.shape) != 2:
        raise ValueError("X_test must be 2D")
    if len(location.shape) != 1:
        raise ValueError("location must be 1D")
    if int(X_test.shape[1]) != int(location.shape[0]):
        raise ValueError("location width must match X_test")
    return AbstractArray(shape=(int(location.shape[0]), int(location.shape[0])), dtype="float64")


def witness_covariance_mahalanobis_location_row(
    location: AbstractArray,
) -> AbstractArray:
    """Describe the singleton-row location matrix passed to pairwise distances."""
    if len(location.shape) != 1:
        raise ValueError("location must be 1D")
    return AbstractArray(shape=(1, int(location.shape[0])), dtype="float64")


def witness_covariance_mahalanobis_result(
    distances: AbstractArray,
) -> AbstractArray:
    """Describe the flattened squared Mahalanobis distance vector."""
    if len(distances.shape) != 2:
        raise ValueError("distances must be 2D")
    if int(distances.shape[1]) != 1:
        raise ValueError("distances must have one column")
    return AbstractArray(shape=(int(distances.shape[0]),), dtype="float64")
