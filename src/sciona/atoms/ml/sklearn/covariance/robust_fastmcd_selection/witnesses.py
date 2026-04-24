"""Ghost witnesses for multivariate FastMCD selection helper atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def _check_positive_int(value: int, name: str) -> int:
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


def _check_vector(values: AbstractArray, name: str) -> int:
    if len(values.shape) != 1:
        raise ValueError(f"{name} must be a vector")
    size = int(values.shape[0])
    if size < 1:
        raise ValueError(f"{name} must be nonempty")
    return size


def _check_matrix(values: AbstractArray, name: str) -> tuple[int, int]:
    if len(values.shape) != 2:
        raise ValueError(f"{name} must be a matrix")
    rows, cols = int(values.shape[0]), int(values.shape[1])
    if rows < 1 or cols < 1:
        raise ValueError(f"{name} must be nonempty")
    return rows, cols


def _check_covariance_stack(values: AbstractArray, name: str) -> tuple[int, int]:
    if len(values.shape) != 3:
        raise ValueError(f"{name} must be a rank-3 tensor")
    depth, rows, cols = map(int, values.shape)
    if depth < 1 or rows < 1 or cols < 1:
        raise ValueError(f"{name} must be nonempty")
    if rows != cols:
        raise ValueError(f"{name} matrices must be square")
    return depth, rows


def witness_fast_mcd_trial_plan(
    n_trials: int | tuple[AbstractArray, AbstractArray],
) -> tuple[bool, int]:
    """Describe whether FastMCD uses supplied estimates and how many starts it runs."""
    if isinstance(n_trials, int):
        _check_positive_int(n_trials, "n_trials")
        return False, 1
    if not isinstance(n_trials, tuple) or len(n_trials) != 2:
        raise TypeError("n_trials must be a positive integer or a tuple of estimate arrays")
    trial_count, n_features = _check_matrix(n_trials[0], "initial_locations")
    covariance_count, covariance_features = _check_covariance_stack(n_trials[1], "initial_covariances")
    if covariance_count != trial_count:
        raise ValueError("initial estimate counts must match")
    if covariance_features != n_features:
        raise ValueError("location and covariance feature dimensions must match")
    return True, 1


def witness_fast_mcd_best_candidate_indices(
    determinants: AbstractArray,
    *,
    select: int = 1,
) -> AbstractArray:
    """Describe the ranked indices of the lowest-determinant candidate estimates."""
    rows = _check_vector(determinants, "determinants")
    _check_positive_int(select, "select")
    if select > rows:
        raise ValueError("select must not exceed the number of determinants")
    return AbstractArray(shape=(select,), dtype="int64")


def witness_fast_mcd_gather_best_candidates(
    locations: AbstractArray,
    covariances: AbstractArray,
    supports: AbstractArray,
    distances: AbstractArray,
    indices: AbstractArray,
) -> tuple[AbstractArray, AbstractArray, AbstractArray, AbstractArray]:
    """Describe gathering the chosen FastMCD candidate tensors by ranked indices."""
    candidate_count, n_features = _check_matrix(locations, "locations")
    covariance_count, covariance_features = _check_covariance_stack(covariances, "covariances")
    support_count, n_samples = _check_matrix(supports, "supports")
    distance_count, distance_samples = _check_matrix(distances, "distances")
    selected = _check_vector(indices, "indices")
    if covariance_count != candidate_count or support_count != candidate_count or distance_count != candidate_count:
        raise ValueError("all candidate tensors must share the same candidate count")
    if covariance_features != n_features:
        raise ValueError("location and covariance feature dimensions must match")
    if distance_samples != n_samples:
        raise ValueError("support and distance sample dimensions must match")
    return (
        AbstractArray(shape=(selected, n_features), dtype="float64"),
        AbstractArray(shape=(selected, n_features, n_features), dtype="float64"),
        AbstractArray(shape=(selected, n_samples), dtype="bool"),
        AbstractArray(shape=(selected, n_samples), dtype="float64"),
    )


def witness_fast_mcd_large_sample_schedule(
    n_samples: int,
    n_features: int,
    n_support: int,
) -> tuple[int, int, int, int, int, int, int, int, int, int]:
    """Describe the deterministic scheduling constants for the large-sample FastMCD branch."""
    _check_positive_int(n_samples, "n_samples")
    _check_positive_int(n_features, "n_features")
    _check_positive_int(n_support, "n_support")
    if n_samples <= 500:
        raise ValueError("n_samples must exceed 500 for the large-sample branch")
    if n_features <= 1:
        raise ValueError("n_features must exceed 1 for the multivariate branch")
    if n_support > n_samples:
        raise ValueError("n_support must not exceed n_samples")
    return (1, 1, 1, 500, 10, 10, 10, 1, 1, 1)


def witness_fast_mcd_place_merged_results(
    n_samples: int,
    selection: AbstractArray,
    merged_support: AbstractArray,
    merged_distances: AbstractArray,
) -> tuple[AbstractArray, AbstractArray]:
    """Describe scattering merged-set support and distances back to the full sample space."""
    _check_positive_int(n_samples, "n_samples")
    selected = _check_vector(selection, "selection")
    support_size = _check_vector(merged_support, "merged_support")
    distance_size = _check_vector(merged_distances, "merged_distances")
    if selected != support_size or selected != distance_size:
        raise ValueError("selection, merged_support, and merged_distances must align")
    return (
        AbstractArray(shape=(n_samples,), dtype="bool"),
        AbstractArray(shape=(n_samples,), dtype="float64"),
    )
