"""Ghost witnesses for sklearn neighbors atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray

from .state_models import NeighborsGraphTransformerState


def _check_matrix(X: AbstractArray) -> tuple[int, int]:
    if len(X.shape) != 2:
        raise ValueError("X must be 2D")
    return int(X.shape[0]), int(X.shape[1])


def _check_minkowski_options(metric: str, p: float, metric_params: None, n_jobs: None) -> None:
    if metric != "minkowski":
        raise ValueError("only minkowski metric is covered")
    if not isinstance(p, (int, float)) or isinstance(p, bool) or p < 1:
        raise ValueError("p must be at least 1")
    if metric_params is not None:
        raise ValueError("metric_params are not covered")
    if n_jobs is not None:
        raise ValueError("parallel jobs are not covered")


def _check_algorithm_options(algorithm: str, leaf_size: int) -> None:
    if algorithm not in {"auto", "brute", "kd_tree", "ball_tree"}:
        raise ValueError("algorithm must be a sklearn neighbor search option")
    if not isinstance(leaf_size, int) or isinstance(leaf_size, bool) or leaf_size < 1:
        raise ValueError("leaf_size must be positive")


def _check_include_self(include_self: bool | str) -> None:
    if not isinstance(include_self, bool) and include_self != "auto":
        raise ValueError("include_self must be bool or auto")


def witness_kneighbors_graph(
    X: AbstractArray,
    n_neighbors: int,
    *,
    mode: str = "connectivity",
    metric: str = "minkowski",
    p: float = 2.0,
    metric_params: None = None,
    include_self: bool | str = False,
    n_jobs: None = None,
) -> AbstractArray:
    """Describe a dense k-neighbor graph."""
    n_samples, _ = _check_matrix(X)
    _check_minkowski_options(metric, p, metric_params, n_jobs)
    _check_include_self(include_self)
    if mode not in {"connectivity", "distance"}:
        raise ValueError("mode must be connectivity or distance")
    if not isinstance(n_neighbors, int) or isinstance(n_neighbors, bool) or n_neighbors < 1:
        raise ValueError("n_neighbors must be positive")
    if n_neighbors > n_samples:
        raise ValueError("n_neighbors must not exceed sample count")
    return AbstractArray(shape=(n_samples, n_samples), dtype="float64")


def witness_radius_neighbors_graph(
    X: AbstractArray,
    radius: float,
    *,
    mode: str = "connectivity",
    metric: str = "minkowski",
    p: float = 2.0,
    metric_params: None = None,
    include_self: bool | str = False,
    n_jobs: None = None,
) -> AbstractArray:
    """Describe a dense radius-neighbor graph."""
    n_samples, _ = _check_matrix(X)
    _check_minkowski_options(metric, p, metric_params, n_jobs)
    _check_include_self(include_self)
    if mode not in {"connectivity", "distance"}:
        raise ValueError("mode must be connectivity or distance")
    if not isinstance(radius, (int, float)) or isinstance(radius, bool) or radius < 0:
        raise ValueError("radius must be nonnegative")
    return AbstractArray(shape=(n_samples, n_samples), dtype="float64")


def witness_kneighbors_transformer_fit(
    X: AbstractArray,
    *,
    mode: str = "distance",
    n_neighbors: int = 5,
    algorithm: str = "auto",
    leaf_size: int = 30,
    metric: str = "minkowski",
    p: float = 2.0,
    metric_params: None = None,
    n_jobs: None = None,
) -> AbstractArray:
    """Describe fitting a dense k-neighbor graph transformer."""
    n_samples, _ = _check_matrix(X)
    _check_algorithm_options(algorithm, leaf_size)
    _check_minkowski_options(metric, p, metric_params, n_jobs)
    if mode not in {"connectivity", "distance"}:
        raise ValueError("mode must be connectivity or distance")
    if not isinstance(n_neighbors, int) or isinstance(n_neighbors, bool) or n_neighbors < 1:
        raise ValueError("n_neighbors must be positive")
    if n_neighbors >= n_samples:
        raise ValueError("n_neighbors must be below sample count")
    return AbstractArray(shape=(n_samples, n_samples), dtype="float64")


def witness_kneighbors_transform(X: AbstractArray, state: NeighborsGraphTransformerState) -> AbstractArray:
    """Describe transforming samples with a fitted dense k-neighbor graph state."""
    n_queries, n_features = _check_matrix(X)
    if n_features != state.n_features_in:
        raise ValueError("X feature count must match fitted state")
    return AbstractArray(shape=(n_queries, state.training_data.shape[0]), dtype="float64")


def witness_radius_neighbors_transformer_fit(
    X: AbstractArray,
    *,
    mode: str = "distance",
    radius: float = 1.0,
    algorithm: str = "auto",
    leaf_size: int = 30,
    metric: str = "minkowski",
    p: float = 2.0,
    metric_params: None = None,
    n_jobs: None = None,
) -> AbstractArray:
    """Describe fitting a dense radius-neighbor graph transformer."""
    n_samples, _ = _check_matrix(X)
    _check_algorithm_options(algorithm, leaf_size)
    _check_minkowski_options(metric, p, metric_params, n_jobs)
    if mode not in {"connectivity", "distance"}:
        raise ValueError("mode must be connectivity or distance")
    if not isinstance(radius, (int, float)) or isinstance(radius, bool) or radius < 0:
        raise ValueError("radius must be nonnegative")
    return AbstractArray(shape=(n_samples, n_samples), dtype="float64")


def witness_radius_neighbors_transform(X: AbstractArray, state: NeighborsGraphTransformerState) -> AbstractArray:
    """Describe transforming samples with a fitted dense radius-neighbor graph state."""
    n_queries, n_features = _check_matrix(X)
    if n_features != state.n_features_in:
        raise ValueError("X feature count must match fitted state")
    return AbstractArray(shape=(n_queries, state.training_data.shape[0]), dtype="float64")
