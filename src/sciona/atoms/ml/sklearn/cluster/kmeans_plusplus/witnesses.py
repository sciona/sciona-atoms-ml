"""Ghost witnesses for dense k-means++ seeding atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def _check_vector(values: AbstractArray, name: str) -> int:
    if len(values.shape) != 1:
        raise ValueError(f"{name} must be 1D")
    size = int(values.shape[0])
    if size < 1:
        raise ValueError(f"{name} must be nonempty")
    return size


def _check_matrix(values: AbstractArray, name: str) -> tuple[int, int]:
    if len(values.shape) != 2:
        raise ValueError(f"{name} must be 2D")
    rows, cols = int(values.shape[0]), int(values.shape[1])
    if rows < 1 or cols < 1:
        raise ValueError(f"{name} must be nonempty")
    return rows, cols


def witness_kmeans_plusplus_default_local_trials(n_clusters: int) -> int:
    """Describe the default greedy-trial count for k-means++ seeding."""
    if n_clusters < 1:
        raise ValueError("n_clusters must be positive")
    return 1


def witness_kmeans_plusplus_first_center_index(sample_weight: AbstractArray, random_state: int | None = None) -> int:
    """Describe choosing the first weighted center index."""
    del random_state
    _check_vector(sample_weight, "sample_weight")
    return 0


def witness_kmeans_plusplus_candidate_ids(
    closest_dist_sq: AbstractArray,
    sample_weight: AbstractArray,
    current_pot: float,
    n_local_trials: int,
    random_state: int | None = None,
) -> AbstractArray:
    """Describe candidate-id sampling for one k-means++ expansion step."""
    del current_pot, random_state
    n_samples = _check_vector(closest_dist_sq, "closest_dist_sq")
    if _check_vector(sample_weight, "sample_weight") != n_samples:
        raise ValueError("sample_weight must match closest_dist_sq")
    if n_local_trials < 1:
        raise ValueError("n_local_trials must be positive")
    return AbstractArray(shape=(n_local_trials,), dtype="int64")


def witness_kmeans_plusplus_candidate_potentials(
    X: AbstractArray,
    x_squared_norms: AbstractArray,
    sample_weight: AbstractArray,
    closest_dist_sq: AbstractArray,
    candidate_ids: AbstractArray,
) -> tuple[AbstractArray, AbstractArray]:
    """Describe candidate distance updates and resulting potentials."""
    n_samples, n_features = _check_matrix(X, "X")
    del n_features
    if _check_vector(x_squared_norms, "x_squared_norms") != n_samples:
        raise ValueError("x_squared_norms must match X")
    if _check_vector(sample_weight, "sample_weight") != n_samples:
        raise ValueError("sample_weight must match X")
    if _check_vector(closest_dist_sq, "closest_dist_sq") != n_samples:
        raise ValueError("closest_dist_sq must match X")
    n_candidates = _check_vector(candidate_ids, "candidate_ids")
    return (
        AbstractArray(shape=(n_candidates, n_samples), dtype="float64"),
        AbstractArray(shape=(n_candidates,), dtype="float64"),
    )


def witness_kmeans_plusplus_initialize_dense(
    X: AbstractArray,
    n_clusters: int,
    sample_weight: AbstractArray | None = None,
    x_squared_norms: AbstractArray | None = None,
    random_state: int | None = None,
    n_local_trials: int | None = None,
) -> tuple[AbstractArray, AbstractArray]:
    """Describe dense k-means++ centers and source indices."""
    del sample_weight, x_squared_norms, random_state, n_local_trials
    n_samples, n_features = _check_matrix(X, "X")
    if n_clusters < 1 or n_clusters > n_samples:
        raise ValueError("n_clusters must be between 1 and the number of samples")
    return (
        AbstractArray(shape=(n_clusters, n_features), dtype="float64"),
        AbstractArray(shape=(n_clusters,), dtype="int64"),
    )
