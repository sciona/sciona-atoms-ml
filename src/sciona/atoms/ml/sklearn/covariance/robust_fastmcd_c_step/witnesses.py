"""Ghost witnesses for FastMCD c-step helper atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_fast_mcd_initial_random_support_indices(
    n_samples: int,
    n_support: int,
    random_state: int | None = None,
) -> AbstractArray:
    """Describe a random initial FastMCD support index vector."""
    del random_state
    if n_samples < 2:
        raise ValueError("n_samples must be at least 2")
    if n_support < 1 or n_support > n_samples:
        raise ValueError("n_support must lie in [1, n_samples]")
    return AbstractArray(shape=(n_support,), dtype="int64", min_val=0)


def witness_fast_mcd_support_indices_from_estimates(
    X: AbstractArray,
    location: AbstractArray,
    covariance: AbstractArray,
    *,
    n_support: int,
) -> AbstractArray:
    """Describe support indices chosen from supplied FastMCD estimates."""
    if len(X.shape) != 2:
        raise ValueError("X must be 2D")
    n_samples, n_features = int(X.shape[0]), int(X.shape[1])
    if len(location.shape) != 1 or int(location.shape[0]) != n_features:
        raise ValueError("location must match the feature count")
    if len(covariance.shape) != 2 or int(covariance.shape[0]) != n_features or int(covariance.shape[1]) != n_features:
        raise ValueError("covariance must be square and match the feature count")
    if n_support < 1 or n_support > n_samples:
        raise ValueError("n_support must lie in [1, n_samples]")
    return AbstractArray(shape=(n_support,), dtype="int64", min_val=0)


def witness_fast_mcd_support_statistics(
    X: AbstractArray,
    support_indices: AbstractArray,
) -> tuple[AbstractArray, AbstractArray, float]:
    """Describe FastMCD support statistics from a support index set."""
    if len(X.shape) != 2:
        raise ValueError("X must be 2D")
    n_features = int(X.shape[1])
    if len(support_indices.shape) != 1 or int(support_indices.shape[0]) < 1:
        raise ValueError("support_indices must be a nonempty 1D array")
    return (
        AbstractArray(shape=(n_features,), dtype="float64"),
        AbstractArray(shape=(n_features, n_features), dtype="float64"),
        0.0,
    )


def witness_fast_mcd_c_step(
    X: AbstractArray,
    n_support: int,
    remaining_iterations: int = 30,
    random_state: int | None = None,
    initial_location: AbstractArray | None = None,
    initial_covariance: AbstractArray | None = None,
) -> tuple[AbstractArray, AbstractArray, float, AbstractArray, AbstractArray]:
    """Describe FastMCD c-step outputs."""
    del random_state
    if len(X.shape) != 2:
        raise ValueError("X must be 2D")
    n_samples, n_features = int(X.shape[0]), int(X.shape[1])
    if n_support < 1 or n_support > n_samples:
        raise ValueError("n_support must lie in [1, n_samples]")
    if remaining_iterations < 0:
        raise ValueError("remaining_iterations must be nonnegative")
    if initial_location is not None and (len(initial_location.shape) != 1 or int(initial_location.shape[0]) != n_features):
        raise ValueError("initial_location must match the feature count")
    if initial_covariance is not None and (len(initial_covariance.shape) != 2 or int(initial_covariance.shape[0]) != n_features or int(initial_covariance.shape[1]) != n_features):
        raise ValueError("initial_covariance must be square and match the feature count")
    return (
        AbstractArray(shape=(n_features,), dtype="float64"),
        AbstractArray(shape=(n_features, n_features), dtype="float64"),
        0.0,
        AbstractArray(shape=(n_samples,), dtype="bool"),
        AbstractArray(shape=(n_samples,), dtype="float64", min_val=0.0),
    )
