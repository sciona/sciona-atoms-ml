"""Ghost witnesses for FastMCD candidate-pool helper atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def _check_data(X: AbstractArray) -> tuple[int, int]:
    if len(X.shape) != 2:
        raise ValueError("X must be 2D")
    n_samples, n_features = map(int, X.shape)
    if n_samples < 2 or n_features < 1:
        raise ValueError("X must be nonempty with at least two samples")
    return n_samples, n_features


def _candidate_pool_shape(trial_count: int, n_samples: int, n_features: int) -> tuple[AbstractArray, AbstractArray, AbstractArray, AbstractArray, AbstractArray]:
    return (
        AbstractArray(shape=(trial_count, n_features), dtype="float64"),
        AbstractArray(shape=(trial_count, n_features, n_features), dtype="float64"),
        AbstractArray(shape=(trial_count,), dtype="float64"),
        AbstractArray(shape=(trial_count, n_samples), dtype="bool"),
        AbstractArray(shape=(trial_count, n_samples), dtype="float64", min_val=0.0),
    )


def witness_fast_mcd_candidate_pool_from_random_starts(
    X: AbstractArray,
    n_support: int,
    n_trials: int,
    n_iter: int = 30,
    random_state: int | None = None,
) -> tuple[AbstractArray, AbstractArray, AbstractArray, AbstractArray, AbstractArray]:
    """Describe a FastMCD candidate pool built from random support starts."""
    del random_state
    n_samples, n_features = _check_data(X)
    if n_support < 1 or n_support > n_samples:
        raise ValueError("n_support must lie in [1, n_samples]")
    if n_trials < 1:
        raise ValueError("n_trials must be positive")
    if n_iter < 0:
        raise ValueError("n_iter must be nonnegative")
    return _candidate_pool_shape(n_trials, n_samples, n_features)


def witness_fast_mcd_candidate_pool_from_estimates(
    X: AbstractArray,
    initial_locations: AbstractArray,
    initial_covariances: AbstractArray,
    n_support: int,
    n_iter: int = 30,
    random_state: int | None = None,
) -> tuple[AbstractArray, AbstractArray, AbstractArray, AbstractArray, AbstractArray]:
    """Describe a FastMCD candidate pool built from supplied estimate stacks."""
    del random_state
    n_samples, n_features = _check_data(X)
    if len(initial_locations.shape) != 2:
        raise ValueError("initial_locations must be 2D")
    trial_count, location_features = map(int, initial_locations.shape)
    if trial_count < 1 or location_features != n_features:
        raise ValueError("initial_locations must be nonempty and match the feature count")
    if len(initial_covariances.shape) != 3:
        raise ValueError("initial_covariances must be 3D")
    cov_trials, cov_rows, cov_cols = map(int, initial_covariances.shape)
    if cov_trials != trial_count or cov_rows != n_features or cov_cols != n_features:
        raise ValueError("initial_covariances must align with initial_locations and be square")
    if n_support < 1 or n_support > n_samples:
        raise ValueError("n_support must lie in [1, n_samples]")
    if n_iter < 0:
        raise ValueError("n_iter must be nonnegative")
    return _candidate_pool_shape(trial_count, n_samples, n_features)
