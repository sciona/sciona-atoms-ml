"""Ghost witnesses for sklearn neighbors atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray

from .state_models import NearestCentroidState, NeighborsGraphTransformerState, NeighborsRegressorState


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


def _check_weights(weights: str) -> None:
    if weights not in {"uniform", "distance"}:
        raise ValueError("weights must be uniform or distance")


def _check_include_self(include_self: bool | str) -> None:
    if not isinstance(include_self, bool) and include_self != "auto":
        raise ValueError("include_self must be bool or auto")


def _check_nearest_centroid_state(X: AbstractArray, state: NearestCentroidState) -> int:
    n_queries, n_features = _check_matrix(X)
    if n_features != state.n_features_in:
        raise ValueError("X feature count must match fitted state")
    if state.classes.shape[0] < 2:
        raise ValueError("state must contain at least two classes")
    return n_queries


def _check_regressor_state(X: AbstractArray, state: NeighborsRegressorState, expected_kind: str) -> tuple[int, int]:
    n_queries, n_features = _check_matrix(X)
    if n_features != state.n_features_in:
        raise ValueError("X feature count must match fitted state")
    if state.regressor_kind != expected_kind:
        raise ValueError("state must match requested neighbor regressor kind")
    _check_weights(state.weights)
    return n_queries, state.target.shape[1]


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


def witness_kneighbors_regressor_fit(
    X: AbstractArray,
    y: AbstractArray,
    *,
    n_neighbors: int = 5,
    weights: str = "uniform",
    algorithm: str = "auto",
    leaf_size: int = 30,
    p: float = 2.0,
    metric: str = "minkowski",
    metric_params: None = None,
    n_jobs: None = None,
) -> AbstractArray:
    """Describe fitting a dense k-neighbor regressor."""
    n_samples, n_features = _check_matrix(X)
    if len(y.shape) not in {1, 2} or y.shape[0] != n_samples:
        raise ValueError("y must be 1D or 2D and match X samples")
    if not isinstance(n_neighbors, int) or isinstance(n_neighbors, bool) or n_neighbors < 1:
        raise ValueError("n_neighbors must be positive")
    if n_neighbors > n_samples:
        raise ValueError("n_neighbors must not exceed sample count")
    _check_weights(weights)
    _check_algorithm_options(algorithm, leaf_size)
    _check_minkowski_options(metric, p, metric_params, n_jobs)
    return AbstractArray(shape=(n_samples, n_features), dtype="float64")


def witness_kneighbors_regressor_predict(X: AbstractArray, state: NeighborsRegressorState) -> AbstractArray:
    """Describe dense k-neighbor regression prediction."""
    n_queries, n_outputs = _check_regressor_state(X, state, "kneighbors")
    if state.outputs_2d:
        return AbstractArray(shape=(n_queries, n_outputs), dtype="float64")
    return AbstractArray(shape=(n_queries,), dtype="float64")


def witness_radius_neighbors_regressor_fit(
    X: AbstractArray,
    y: AbstractArray,
    *,
    radius: float = 1.0,
    weights: str = "uniform",
    algorithm: str = "auto",
    leaf_size: int = 30,
    p: float = 2.0,
    metric: str = "minkowski",
    metric_params: None = None,
    n_jobs: None = None,
) -> AbstractArray:
    """Describe fitting a dense radius-neighbor regressor."""
    n_samples, n_features = _check_matrix(X)
    if len(y.shape) not in {1, 2} or y.shape[0] != n_samples:
        raise ValueError("y must be 1D or 2D and match X samples")
    if not isinstance(radius, (int, float)) or isinstance(radius, bool) or radius < 0:
        raise ValueError("radius must be nonnegative")
    _check_weights(weights)
    _check_algorithm_options(algorithm, leaf_size)
    _check_minkowski_options(metric, p, metric_params, n_jobs)
    return AbstractArray(shape=(n_samples, n_features), dtype="float64")


def witness_radius_neighbors_regressor_predict(X: AbstractArray, state: NeighborsRegressorState) -> AbstractArray:
    """Describe dense radius-neighbor regression prediction."""
    n_queries, n_outputs = _check_regressor_state(X, state, "radius_neighbors")
    if state.outputs_2d:
        return AbstractArray(shape=(n_queries, n_outputs), dtype="float64")
    return AbstractArray(shape=(n_queries,), dtype="float64")


def witness_nearest_centroid_fit(
    X: AbstractArray,
    y: AbstractArray,
    *,
    metric: str = "euclidean",
    shrink_threshold: float | None = None,
    priors: str | tuple[float, ...] = "uniform",
) -> AbstractArray:
    """Describe fitting a dense nearest-centroid classifier."""
    n_samples, n_features = _check_matrix(X)
    if len(y.shape) != 1 or y.shape[0] != n_samples:
        raise ValueError("y must be 1D and match X samples")
    if metric not in {"euclidean", "manhattan"}:
        raise ValueError("metric must be euclidean or manhattan")
    if shrink_threshold is not None and shrink_threshold <= 0:
        raise ValueError("shrink_threshold must be positive when provided")
    if not isinstance(priors, tuple) and priors not in {"uniform", "empirical"}:
        raise ValueError("priors must be uniform, empirical, or a tuple")
    return AbstractArray(shape=(n_samples, n_features), dtype="float64")


def witness_nearest_centroid_predict(X: AbstractArray, state: NearestCentroidState) -> AbstractArray:
    """Describe dense nearest-centroid class prediction."""
    n_queries = _check_nearest_centroid_state(X, state)
    return AbstractArray(shape=(n_queries,), dtype="float64")


def witness_nearest_centroid_decision_function(X: AbstractArray, state: NearestCentroidState) -> AbstractArray:
    """Describe dense nearest-centroid discriminant scores."""
    n_queries = _check_nearest_centroid_state(X, state)
    if state.metric != "euclidean":
        raise ValueError("decision scores are covered for euclidean metric only")
    if state.classes.shape[0] == 2:
        return AbstractArray(shape=(n_queries,), dtype="float64")
    return AbstractArray(shape=(n_queries, state.classes.shape[0]), dtype="float64")


def witness_nearest_centroid_predict_proba(X: AbstractArray, state: NearestCentroidState) -> AbstractArray:
    """Describe dense nearest-centroid class probabilities."""
    n_queries = _check_nearest_centroid_state(X, state)
    if state.metric != "euclidean":
        raise ValueError("probabilities are covered for euclidean metric only")
    return AbstractArray(shape=(n_queries, state.classes.shape[0]), dtype="float64")


def witness_nearest_centroid_predict_log_proba(X: AbstractArray, state: NearestCentroidState) -> AbstractArray:
    """Describe dense nearest-centroid log class probabilities."""
    n_queries = _check_nearest_centroid_state(X, state)
    if state.metric != "euclidean":
        raise ValueError("log probabilities are covered for euclidean metric only")
    return AbstractArray(shape=(n_queries, state.classes.shape[0]), dtype="float64")
