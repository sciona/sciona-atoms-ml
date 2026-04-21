"""Dense neighbors graph atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray
from sklearn.utils.validation import check_X_y, check_array

from sciona.ghost.registry import register_atom

from .state_models import NearestCentroidState, NeighborsGraphTransformerState, NeighborsRegressorState
from .witnesses import (
    witness_kneighbors_graph,
    witness_kneighbors_regressor_fit,
    witness_kneighbors_regressor_predict,
    witness_kneighbors_transform,
    witness_kneighbors_transformer_fit,
    witness_nearest_centroid_decision_function,
    witness_nearest_centroid_fit,
    witness_nearest_centroid_predict,
    witness_nearest_centroid_predict_log_proba,
    witness_nearest_centroid_predict_proba,
    witness_radius_neighbors_graph,
    witness_radius_neighbors_regressor_fit,
    witness_radius_neighbors_regressor_predict,
    witness_radius_neighbors_transform,
    witness_radius_neighbors_transformer_fit,
)


def _matrix_2d(X: NDArray[np.float64]) -> bool:
    return bool(np.asarray(X).ndim == 2)


def _finite_matrix(X: NDArray[np.float64]) -> bool:
    try:
        values = np.asarray(X, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(np.all(np.isfinite(values)))


def _mode_valid(mode: str) -> bool:
    return mode in {"connectivity", "distance"}


def _minkowski_options_valid(metric: str, p: float, metric_params: None, n_jobs: None) -> bool:
    return bool(
        metric == "minkowski"
        and isinstance(p, (int, float))
        and not isinstance(p, bool)
        and np.isfinite(float(p))
        and float(p) >= 1.0
        and metric_params is None
        and n_jobs is None
    )


def _include_self_valid(include_self: bool | str) -> bool:
    return bool(isinstance(include_self, bool) or include_self == "auto")


def _positive_neighbors(n_neighbors: int, X: NDArray[np.float64]) -> bool:
    values = np.asarray(X)
    return bool(
        isinstance(n_neighbors, int)
        and not isinstance(n_neighbors, bool)
        and values.ndim == 2
        and 1 <= n_neighbors <= values.shape[0]
    )


def _positive_neighbors_below_samples(n_neighbors: int, X: NDArray[np.float64]) -> bool:
    values = np.asarray(X)
    return bool(
        isinstance(n_neighbors, int)
        and not isinstance(n_neighbors, bool)
        and values.ndim == 2
        and 1 <= n_neighbors < values.shape[0]
    )


def _radius_valid(radius: float) -> bool:
    return bool(
        isinstance(radius, (int, float))
        and not isinstance(radius, bool)
        and np.isfinite(float(radius))
        and float(radius) >= 0.0
    )


def _algorithm_options_valid(algorithm: str, leaf_size: int) -> bool:
    return bool(
        algorithm in {"auto", "brute", "kd_tree", "ball_tree"}
        and isinstance(leaf_size, int)
        and not isinstance(leaf_size, bool)
        and leaf_size >= 1
    )


def _graph_valid(result: NDArray[np.float64], n_rows: int, n_cols: int) -> bool:
    values = np.asarray(result, dtype=np.float64)
    return bool(values.shape == (n_rows, n_cols) and np.all(np.isfinite(values)) and np.all(values >= 0.0))


def _state_valid(state: NeighborsGraphTransformerState) -> bool:
    return bool(
        state.training_data.ndim == 2
        and state.training_data.shape[1] == state.n_features_in
        and state.mode in {"connectivity", "distance"}
        and state.transformer_kind in {"kneighbors", "radius_neighbors"}
        and state.metric == "minkowski"
        and np.isfinite(state.p)
        and state.p >= 1.0
        and np.all(np.isfinite(state.training_data))
        and (
            (
                state.transformer_kind == "kneighbors"
                and state.n_neighbors is not None
                and 1 <= state.n_neighbors < state.training_data.shape[0]
                and state.radius is None
            )
            or (
                state.transformer_kind == "radius_neighbors"
                and state.n_neighbors is None
                and state.radius is not None
                and np.isfinite(state.radius)
                and state.radius >= 0.0
            )
        )
    )


def _feature_count_matches(X: NDArray[np.float64], state: NeighborsGraphTransformerState) -> bool:
    values = np.asarray(X)
    return bool(values.ndim == 2 and values.shape[1] == state.n_features_in)


def _target_1d(y: NDArray[np.float64]) -> bool:
    return bool(np.asarray(y).ndim == 1)


def _sample_counts_match(X: NDArray[np.float64], y: NDArray[np.float64]) -> bool:
    values_x = np.asarray(X)
    values_y = np.asarray(y)
    return bool(values_x.ndim == 2 and values_y.ndim == 1 and values_x.shape[0] == values_y.shape[0])


def _target_1d_or_2d(y: NDArray[np.float64]) -> bool:
    return bool(np.asarray(y).ndim in {1, 2})


def _sample_counts_match_regression(X: NDArray[np.float64], y: NDArray[np.float64]) -> bool:
    values_x = np.asarray(X)
    values_y = np.asarray(y)
    return bool(values_x.ndim == 2 and values_y.ndim in {1, 2} and values_x.shape[0] == values_y.shape[0])


def _finite_inputs(X: NDArray[np.float64], y: NDArray[np.float64]) -> bool:
    try:
        values_x = np.asarray(X, dtype=np.float64)
        values_y = np.asarray(y, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(np.all(np.isfinite(values_x)) and np.all(np.isfinite(values_y)))


def _weights_valid(weights: str) -> bool:
    return weights in {"uniform", "distance"}


def _at_least_two_classes(y: NDArray[np.float64]) -> bool:
    return bool(np.unique(np.asarray(y, dtype=np.float64)).shape[0] >= 2)


def _sample_count_exceeds_class_count(X: NDArray[np.float64], y: NDArray[np.float64]) -> bool:
    values_x = np.asarray(X)
    values_y = np.asarray(y)
    return bool(values_x.ndim == 2 and values_y.ndim == 1 and values_x.shape[0] > np.unique(values_y).shape[0])


def _nearest_centroid_options_valid(
    metric: str,
    shrink_threshold: float | None,
    priors: str | tuple[float, ...],
    y: NDArray[np.float64],
) -> bool:
    if metric not in {"euclidean", "manhattan"}:
        return False
    if shrink_threshold is not None and (
        not isinstance(shrink_threshold, (int, float))
        or isinstance(shrink_threshold, bool)
        or not np.isfinite(float(shrink_threshold))
        or float(shrink_threshold) <= 0.0
    ):
        return False
    n_classes = np.unique(np.asarray(y, dtype=np.float64)).shape[0]
    if priors in {"uniform", "empirical"}:
        return True
    if not isinstance(priors, tuple):
        return False
    values = np.asarray(priors, dtype=np.float64)
    return bool(values.ndim == 1 and values.shape[0] == n_classes and np.all(np.isfinite(values)) and np.all(values >= 0.0) and values.sum() > 0.0)


def _class_centroids(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    classes: NDArray[np.float64],
    metric: str,
) -> NDArray[np.float64]:
    centroids = np.empty((classes.shape[0], X.shape[1]), dtype=np.float64)
    for class_index, class_label in enumerate(classes):
        class_rows = X[y == class_label]
        if metric == "manhattan":
            centroids[class_index] = np.median(class_rows, axis=0)
        else:
            centroids[class_index] = np.mean(class_rows, axis=0)
    return centroids


def _nearest_centroid_denominators_positive(X: NDArray[np.float64], y: NDArray[np.float64], metric: str) -> bool:
    if metric not in {"euclidean", "manhattan"}:
        return False
    values_x = np.asarray(X, dtype=np.float64)
    values_y = np.asarray(y, dtype=np.float64)
    classes = np.unique(values_y)
    if values_x.shape[0] <= classes.shape[0] or np.all(np.ptp(values_x, axis=0) == 0.0):
        return False
    centroids = _class_centroids(values_x, values_y, classes, metric)
    y_ind = np.searchsorted(classes, values_y)
    variance = (values_x - centroids[y_ind]) ** 2
    within_std = np.sqrt(variance.sum(axis=0) / (values_x.shape[0] - classes.shape[0]))
    return bool(np.all(within_std + np.median(within_std) > 0.0))


def _nearest_centroid_state_valid(state: NearestCentroidState) -> bool:
    n_classes = state.classes.shape[0]
    return bool(
        n_classes >= 2
        and state.classes.ndim == 1
        and state.centroids.shape == (n_classes, state.n_features_in)
        and state.deviations.shape == (n_classes, state.n_features_in)
        and state.within_class_std_dev.shape == (state.n_features_in,)
        and state.class_prior.shape == (n_classes,)
        and state.metric in {"euclidean", "manhattan"}
        and (state.shrink_threshold is None or state.shrink_threshold > 0.0)
        and np.all(np.isfinite(state.classes))
        and np.all(np.isfinite(state.centroids))
        and np.all(np.isfinite(state.deviations))
        and np.all(np.isfinite(state.within_class_std_dev))
        and np.all(np.isfinite(state.class_prior))
        and np.all(state.class_prior >= 0.0)
        and np.isclose(np.sum(state.class_prior), 1.0)
    )


def _nearest_centroid_feature_count_matches(X: NDArray[np.float64], state: NearestCentroidState) -> bool:
    values = np.asarray(X)
    return bool(values.ndim == 2 and values.shape[1] == state.n_features_in)


def _nearest_centroid_prediction_valid(result: NDArray[np.float64], X: NDArray[np.float64]) -> bool:
    values = np.asarray(result)
    return bool(values.ndim == 1 and values.shape[0] == np.asarray(X).shape[0] and np.all(np.isfinite(values)))


def _nearest_centroid_scores_valid(result: NDArray[np.float64], X: NDArray[np.float64], state: NearestCentroidState) -> bool:
    values = np.asarray(result)
    if state.classes.shape[0] == 2:
        return bool(values.ndim == 1 and values.shape[0] == np.asarray(X).shape[0] and np.all(np.isfinite(values)))
    return bool(values.shape == (np.asarray(X).shape[0], state.classes.shape[0]) and np.all(np.isfinite(values)))


def _nearest_centroid_proba_valid(result: NDArray[np.float64], X: NDArray[np.float64], state: NearestCentroidState) -> bool:
    values = np.asarray(result)
    return bool(
        values.shape == (np.asarray(X).shape[0], state.classes.shape[0])
        and np.all(np.isfinite(values))
        and np.all(values >= 0.0)
        and np.allclose(values.sum(axis=1), 1.0)
    )


def _nearest_centroid_log_proba_valid(result: NDArray[np.float64], X: NDArray[np.float64], state: NearestCentroidState) -> bool:
    values = np.asarray(result)
    return bool(
        values.shape == (np.asarray(X).shape[0], state.classes.shape[0])
        and np.all(np.isfinite(values))
        and np.allclose(np.exp(values).sum(axis=1), 1.0)
    )


def _regressor_state_valid(state: NeighborsRegressorState) -> bool:
    n_samples = state.training_data.shape[0]
    return bool(
        state.training_data.ndim == 2
        and state.training_data.shape[1] == state.n_features_in
        and state.target.ndim == 2
        and state.target.shape[0] == n_samples
        and state.target.shape[1] >= 1
        and state.weights in {"uniform", "distance"}
        and state.metric == "minkowski"
        and np.isfinite(state.p)
        and state.p >= 1.0
        and state.regressor_kind in {"kneighbors", "radius_neighbors"}
        and isinstance(state.outputs_2d, bool)
        and np.all(np.isfinite(state.training_data))
        and np.all(np.isfinite(state.target))
        and (
            (
                state.regressor_kind == "kneighbors"
                and state.n_neighbors is not None
                and 1 <= state.n_neighbors <= n_samples
                and state.radius is None
            )
            or (
                state.regressor_kind == "radius_neighbors"
                and state.n_neighbors is None
                and state.radius is not None
                and np.isfinite(state.radius)
                and state.radius >= 0.0
            )
        )
    )


def _regressor_feature_count_matches(X: NDArray[np.float64], state: NeighborsRegressorState) -> bool:
    values = np.asarray(X)
    return bool(values.ndim == 2 and values.shape[1] == state.n_features_in)


def _regressor_prediction_valid(result: NDArray[np.float64], X: NDArray[np.float64], state: NeighborsRegressorState) -> bool:
    values = np.asarray(result)
    n_queries = np.asarray(X).shape[0]
    if state.outputs_2d:
        return bool(values.shape == (n_queries, state.target.shape[1]) and np.all(np.isfinite(values)))
    return bool(values.shape == (n_queries,) and np.all(np.isfinite(values)))


def _radius_regressor_queries_have_neighbors(X: NDArray[np.float64], state: NeighborsRegressorState) -> bool:
    try:
        values = np.asarray(X, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    if not _regressor_state_valid(state) or state.regressor_kind != "radius_neighbors" or values.ndim != 2:
        return False
    if values.shape[1] != state.n_features_in:
        return False
    distances = _pairwise_minkowski(values, state.training_data, state.p)
    return bool(np.all(np.any(distances <= float(state.radius), axis=1)))


def _resolve_include_self(include_self: bool | str, mode: str) -> bool:
    if include_self == "auto":
        return mode == "connectivity"
    return bool(include_self)


def _pairwise_minkowski(X: NDArray[np.float64], Y: NDArray[np.float64], p: float) -> NDArray[np.float64]:
    x_values = np.asarray(X, dtype=np.float64)
    y_values = np.asarray(Y, dtype=np.float64)
    diff = np.abs(x_values[:, np.newaxis, :] - y_values[np.newaxis, :, :])
    if float(p) == 1.0:
        return np.asarray(np.sum(diff, axis=2), dtype=np.float64)
    if float(p) == 2.0:
        return np.asarray(np.sqrt(np.sum(diff * diff, axis=2)), dtype=np.float64)
    return np.asarray(np.sum(diff**float(p), axis=2) ** (1.0 / float(p)), dtype=np.float64)


def _fill_kneighbor_graph(
    distances: NDArray[np.float64],
    n_neighbors: int,
    mode: str,
    *,
    exclude_diagonal: bool,
) -> NDArray[np.float64]:
    graph = np.zeros_like(distances, dtype=np.float64)
    ranking_distances = np.asarray(distances, dtype=np.float64).copy()
    if exclude_diagonal and ranking_distances.shape[0] == ranking_distances.shape[1]:
        ranking_distances.flat[:: ranking_distances.shape[0] + 1] = np.inf
    order = np.argsort(ranking_distances, axis=1, kind="stable")[:, :n_neighbors]
    rows = np.arange(distances.shape[0])[:, np.newaxis]
    if mode == "connectivity":
        graph[rows, order] = 1.0
    else:
        graph[rows, order] = distances[rows, order]
    return graph


def _fill_radius_graph(
    distances: NDArray[np.float64],
    radius: float,
    mode: str,
    *,
    exclude_diagonal: bool,
) -> NDArray[np.float64]:
    graph = np.zeros_like(distances, dtype=np.float64)
    mask = distances <= float(radius)
    if exclude_diagonal and mask.shape[0] == mask.shape[1]:
        mask.flat[:: mask.shape[0] + 1] = False
    if mode == "connectivity":
        graph[mask] = 1.0
    else:
        graph[mask] = distances[mask]
    return graph


def _checked_regression_inputs(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], bool]:
    checked_x, checked_y = check_X_y(X, y, dtype=np.float64, multi_output=True, y_numeric=True)
    target = np.asarray(checked_y, dtype=np.float64)
    outputs_2d = target.ndim == 2
    if not outputs_2d:
        target = target.reshape(-1, 1)
    return np.asarray(checked_x, dtype=np.float64), target, outputs_2d


def _kneighbor_indices_and_distances(
    X: NDArray[np.float64],
    state: NeighborsRegressorState,
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    distances = _pairwise_minkowski(X, state.training_data, state.p)
    order = np.argsort(distances, axis=1, kind="stable")[:, : int(state.n_neighbors or 0)]
    rows = np.arange(distances.shape[0])[:, np.newaxis]
    return np.asarray(order, dtype=np.int64), np.asarray(distances[rows, order], dtype=np.float64)


def _distance_weight_matrix(distances: NDArray[np.float64]) -> NDArray[np.float64]:
    with np.errstate(divide="ignore"):
        weights = 1.0 / distances
    inf_mask = np.isinf(weights)
    inf_rows = np.any(inf_mask, axis=1)
    weights[inf_rows] = inf_mask[inf_rows].astype(np.float64)
    return np.asarray(weights, dtype=np.float64)


def _distance_weight_vector(distances: NDArray[np.float64]) -> NDArray[np.float64]:
    with np.errstate(divide="ignore"):
        weights = 1.0 / distances
    if np.any(np.isinf(weights)):
        return np.asarray(np.isinf(weights), dtype=np.float64)
    return np.asarray(weights, dtype=np.float64)


def _weighted_regression_targets(
    targets: NDArray[np.float64],
    indices: NDArray[np.int64],
    weights: NDArray[np.float64],
) -> NDArray[np.float64]:
    numerator = np.sum(targets[indices] * weights[:, :, np.newaxis], axis=1)
    denominator = np.sum(weights, axis=1)[:, np.newaxis]
    return np.asarray(numerator / denominator, dtype=np.float64)


def _nearest_centroid_distances(
    X: NDArray[np.float64],
    centroids: NDArray[np.float64],
    metric: str,
) -> NDArray[np.float64]:
    if metric == "manhattan":
        return np.asarray(np.sum(np.abs(X[:, np.newaxis, :] - centroids[np.newaxis, :, :]), axis=2), dtype=np.float64)
    diff = X[:, np.newaxis, :] - centroids[np.newaxis, :, :]
    return np.asarray(np.sqrt(np.sum(diff * diff, axis=2)), dtype=np.float64)


def _nearest_centroid_raw_scores(X: NDArray[np.float64], state: NearestCentroidState) -> NDArray[np.float64]:
    x_normalized = np.asarray(X, dtype=np.float64).copy()
    mask = state.within_class_std_dev != 0.0
    x_normalized[:, mask] /= state.within_class_std_dev[mask]
    centroids_normalized = state.centroids.copy()
    centroids_normalized[:, mask] /= state.within_class_std_dev[mask]
    distances = _nearest_centroid_distances(x_normalized, centroids_normalized, state.metric)
    return np.asarray(-(distances**2) + 2.0 * np.log(state.class_prior[np.newaxis, :]), dtype=np.float64)


def _softmax(scores: NDArray[np.float64]) -> NDArray[np.float64]:
    shifted = scores - np.max(scores, axis=1)[:, np.newaxis]
    exponent = np.exp(shifted)
    return np.asarray(exponent / exponent.sum(axis=1)[:, np.newaxis], dtype=np.float64)


@register_atom(witness_kneighbors_graph)
@icontract.require(lambda X: _matrix_2d(X), "X must be 2D")
@icontract.require(lambda X: _finite_matrix(X), "X must contain finite values")
@icontract.require(lambda n_neighbors, X: _positive_neighbors(n_neighbors, X), "n_neighbors must fit sample count")
@icontract.require(lambda mode: _mode_valid(mode), "mode must be connectivity or distance")
@icontract.require(lambda metric, p, metric_params, n_jobs: _minkowski_options_valid(metric, p, metric_params, n_jobs), "only dense minkowski search is covered")
@icontract.require(lambda include_self: _include_self_valid(include_self), "include_self must be bool or auto")
@icontract.ensure(lambda result, X: _graph_valid(result, np.asarray(X).shape[0], np.asarray(X).shape[0]), "graph must be finite and nonnegative")
def kneighbors_graph(
    X: NDArray[np.float64],
    n_neighbors: int,
    *,
    mode: str = "connectivity",
    metric: str = "minkowski",
    p: float = 2.0,
    metric_params: None = None,
    include_self: bool | str = False,
    n_jobs: None = None,
) -> NDArray[np.float64]:
    """Compute a dense k-neighbor connectivity or distance graph."""
    checked = check_array(X, dtype=np.float64, ensure_2d=True)
    include = _resolve_include_self(include_self, mode)
    if not include and n_neighbors >= checked.shape[0]:
        raise ValueError("n_neighbors must be below sample count when self-neighbors are excluded")
    distances = _pairwise_minkowski(checked, checked, float(p))
    return _fill_kneighbor_graph(distances, n_neighbors, mode, exclude_diagonal=not include)


@register_atom(witness_radius_neighbors_graph)
@icontract.require(lambda X: _matrix_2d(X), "X must be 2D")
@icontract.require(lambda X: _finite_matrix(X), "X must contain finite values")
@icontract.require(lambda radius: _radius_valid(radius), "radius must be nonnegative and finite")
@icontract.require(lambda mode: _mode_valid(mode), "mode must be connectivity or distance")
@icontract.require(lambda metric, p, metric_params, n_jobs: _minkowski_options_valid(metric, p, metric_params, n_jobs), "only dense minkowski search is covered")
@icontract.require(lambda include_self: _include_self_valid(include_self), "include_self must be bool or auto")
@icontract.ensure(lambda result, X: _graph_valid(result, np.asarray(X).shape[0], np.asarray(X).shape[0]), "graph must be finite and nonnegative")
def radius_neighbors_graph(
    X: NDArray[np.float64],
    radius: float,
    *,
    mode: str = "connectivity",
    metric: str = "minkowski",
    p: float = 2.0,
    metric_params: None = None,
    include_self: bool | str = False,
    n_jobs: None = None,
) -> NDArray[np.float64]:
    """Compute a dense radius-neighbor connectivity or distance graph."""
    checked = check_array(X, dtype=np.float64, ensure_2d=True)
    include = _resolve_include_self(include_self, mode)
    distances = _pairwise_minkowski(checked, checked, float(p))
    return _fill_radius_graph(distances, float(radius), mode, exclude_diagonal=not include)


@register_atom(witness_kneighbors_transformer_fit)
@icontract.require(lambda X: _matrix_2d(X), "X must be 2D")
@icontract.require(lambda X: _finite_matrix(X), "X must contain finite values")
@icontract.require(lambda mode: _mode_valid(mode), "mode must be connectivity or distance")
@icontract.require(lambda n_neighbors, X: _positive_neighbors_below_samples(n_neighbors, X), "n_neighbors must be below sample count")
@icontract.require(lambda algorithm, leaf_size: _algorithm_options_valid(algorithm, leaf_size), "algorithm and leaf_size must be valid")
@icontract.require(lambda metric, p, metric_params, n_jobs: _minkowski_options_valid(metric, p, metric_params, n_jobs), "only dense minkowski search is covered")
@icontract.ensure(lambda result: _state_valid(result), "state must contain finite dense k-neighbor transformer data")
def kneighbors_transformer_fit(
    X: NDArray[np.float64],
    *,
    mode: str = "distance",
    n_neighbors: int = 5,
    algorithm: str = "auto",
    leaf_size: int = 30,
    metric: str = "minkowski",
    p: float = 2.0,
    metric_params: None = None,
    n_jobs: None = None,
) -> NeighborsGraphTransformerState:
    """Fit a dense k-neighbor graph transformer state."""
    del algorithm, leaf_size, metric_params, n_jobs
    checked = check_array(X, dtype=np.float64, ensure_2d=True)
    return NeighborsGraphTransformerState(
        training_data=np.asarray(checked, dtype=np.float64).copy(),
        mode=mode,
        n_neighbors=int(n_neighbors),
        radius=None,
        metric=metric,
        p=float(p),
        transformer_kind="kneighbors",
        n_features_in=int(checked.shape[1]),
    )


@register_atom(witness_kneighbors_transform)
@icontract.require(lambda X: _matrix_2d(X), "X must be 2D")
@icontract.require(lambda X: _finite_matrix(X), "X must contain finite values")
@icontract.require(lambda state: _state_valid(state), "state must be a fitted dense k-neighbor transformer")
@icontract.require(lambda state: state.transformer_kind == "kneighbors", "state must be a k-neighbor transformer")
@icontract.require(lambda X, state: _feature_count_matches(X, state), "X feature count must match fitted state")
@icontract.ensure(lambda result, X, state: _graph_valid(result, np.asarray(X).shape[0], state.training_data.shape[0]), "graph must be finite and nonnegative")
def kneighbors_transform(X: NDArray[np.float64], state: NeighborsGraphTransformerState) -> NDArray[np.float64]:
    """Transform samples into a dense k-neighbor graph against fitted data."""
    checked = check_array(X, dtype=np.float64, ensure_2d=True)
    distances = _pairwise_minkowski(checked, state.training_data, state.p)
    add_one = 1 if state.mode == "distance" else 0
    n_neighbors = int(state.n_neighbors or 0) + add_one
    if n_neighbors > state.training_data.shape[0]:
        raise ValueError("effective n_neighbors exceeds fitted sample count")
    return _fill_kneighbor_graph(distances, n_neighbors, state.mode, exclude_diagonal=False)


@register_atom(witness_radius_neighbors_transformer_fit)
@icontract.require(lambda X: _matrix_2d(X), "X must be 2D")
@icontract.require(lambda X: _finite_matrix(X), "X must contain finite values")
@icontract.require(lambda mode: _mode_valid(mode), "mode must be connectivity or distance")
@icontract.require(lambda radius: _radius_valid(radius), "radius must be nonnegative and finite")
@icontract.require(lambda algorithm, leaf_size: _algorithm_options_valid(algorithm, leaf_size), "algorithm and leaf_size must be valid")
@icontract.require(lambda metric, p, metric_params, n_jobs: _minkowski_options_valid(metric, p, metric_params, n_jobs), "only dense minkowski search is covered")
@icontract.ensure(lambda result: _state_valid(result), "state must contain finite dense radius-neighbor transformer data")
def radius_neighbors_transformer_fit(
    X: NDArray[np.float64],
    *,
    mode: str = "distance",
    radius: float = 1.0,
    algorithm: str = "auto",
    leaf_size: int = 30,
    metric: str = "minkowski",
    p: float = 2.0,
    metric_params: None = None,
    n_jobs: None = None,
) -> NeighborsGraphTransformerState:
    """Fit a dense radius-neighbor graph transformer state."""
    del algorithm, leaf_size, metric_params, n_jobs
    checked = check_array(X, dtype=np.float64, ensure_2d=True)
    return NeighborsGraphTransformerState(
        training_data=np.asarray(checked, dtype=np.float64).copy(),
        mode=mode,
        n_neighbors=None,
        radius=float(radius),
        metric=metric,
        p=float(p),
        transformer_kind="radius_neighbors",
        n_features_in=int(checked.shape[1]),
    )


@register_atom(witness_radius_neighbors_transform)
@icontract.require(lambda X: _matrix_2d(X), "X must be 2D")
@icontract.require(lambda X: _finite_matrix(X), "X must contain finite values")
@icontract.require(lambda state: _state_valid(state), "state must be a fitted dense radius-neighbor transformer")
@icontract.require(lambda state: state.transformer_kind == "radius_neighbors", "state must be a radius-neighbor transformer")
@icontract.require(lambda X, state: _feature_count_matches(X, state), "X feature count must match fitted state")
@icontract.ensure(lambda result, X, state: _graph_valid(result, np.asarray(X).shape[0], state.training_data.shape[0]), "graph must be finite and nonnegative")
def radius_neighbors_transform(X: NDArray[np.float64], state: NeighborsGraphTransformerState) -> NDArray[np.float64]:
    """Transform samples into a dense radius-neighbor graph against fitted data."""
    checked = check_array(X, dtype=np.float64, ensure_2d=True)
    distances = _pairwise_minkowski(checked, state.training_data, state.p)
    return _fill_radius_graph(distances, float(state.radius), state.mode, exclude_diagonal=False)


@register_atom(witness_kneighbors_regressor_fit)
@icontract.require(lambda X: _matrix_2d(X), "X must be 2D")
@icontract.require(lambda y: _target_1d_or_2d(y), "y must be 1D or 2D")
@icontract.require(lambda X, y: _sample_counts_match_regression(X, y), "X and y must have matching sample counts")
@icontract.require(lambda X, y: _finite_inputs(X, y), "X and y must be finite numeric arrays")
@icontract.require(lambda n_neighbors, X: _positive_neighbors(n_neighbors, X), "n_neighbors must fit sample count")
@icontract.require(lambda weights: _weights_valid(weights), "weights must be uniform or distance")
@icontract.require(lambda algorithm, leaf_size: _algorithm_options_valid(algorithm, leaf_size), "algorithm and leaf_size must be valid")
@icontract.require(lambda metric, p, metric_params, n_jobs: _minkowski_options_valid(metric, p, metric_params, n_jobs), "only dense minkowski search is covered")
@icontract.ensure(lambda result: _regressor_state_valid(result), "state must contain finite dense k-neighbor regression data")
def kneighbors_regressor_fit(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    *,
    n_neighbors: int = 5,
    weights: str = "uniform",
    algorithm: str = "auto",
    leaf_size: int = 30,
    p: float = 2.0,
    metric: str = "minkowski",
    metric_params: None = None,
    n_jobs: None = None,
) -> NeighborsRegressorState:
    """Fit dense k-neighbor regression state for finite numeric targets."""
    del algorithm, leaf_size, metric_params, n_jobs
    checked_x, target, outputs_2d = _checked_regression_inputs(X, y)
    return NeighborsRegressorState(
        training_data=checked_x.copy(),
        target=target.copy(),
        weights=weights,
        n_neighbors=int(n_neighbors),
        radius=None,
        metric=metric,
        p=float(p),
        regressor_kind="kneighbors",
        n_features_in=int(checked_x.shape[1]),
        outputs_2d=outputs_2d,
    )


@register_atom(witness_kneighbors_regressor_predict)
@icontract.require(lambda X: _matrix_2d(X), "X must be 2D")
@icontract.require(lambda X: _finite_matrix(X), "X must contain finite values")
@icontract.require(lambda state: _regressor_state_valid(state), "state must be a fitted dense k-neighbor regressor")
@icontract.require(lambda state: state.regressor_kind == "kneighbors", "state must be a k-neighbor regressor")
@icontract.require(lambda X, state: _regressor_feature_count_matches(X, state), "X feature count must match fitted state")
@icontract.ensure(lambda result, X, state: _regressor_prediction_valid(result, X, state), "predictions must be finite numeric targets")
def kneighbors_regressor_predict(X: NDArray[np.float64], state: NeighborsRegressorState) -> NDArray[np.float64]:
    """Predict finite numeric targets with dense k-neighbor interpolation."""
    checked = check_array(X, dtype=np.float64, ensure_2d=True)
    neighbor_indices, neighbor_distances = _kneighbor_indices_and_distances(checked, state)
    if state.weights == "uniform":
        y_pred = np.asarray(np.mean(state.target[neighbor_indices], axis=1), dtype=np.float64)
    else:
        y_pred = _weighted_regression_targets(state.target, neighbor_indices, _distance_weight_matrix(neighbor_distances))
    if state.outputs_2d:
        return y_pred
    return np.asarray(y_pred.ravel(), dtype=np.float64)


@register_atom(witness_radius_neighbors_regressor_fit)
@icontract.require(lambda X: _matrix_2d(X), "X must be 2D")
@icontract.require(lambda y: _target_1d_or_2d(y), "y must be 1D or 2D")
@icontract.require(lambda X, y: _sample_counts_match_regression(X, y), "X and y must have matching sample counts")
@icontract.require(lambda X, y: _finite_inputs(X, y), "X and y must be finite numeric arrays")
@icontract.require(lambda radius: _radius_valid(radius), "radius must be nonnegative and finite")
@icontract.require(lambda weights: _weights_valid(weights), "weights must be uniform or distance")
@icontract.require(lambda algorithm, leaf_size: _algorithm_options_valid(algorithm, leaf_size), "algorithm and leaf_size must be valid")
@icontract.require(lambda metric, p, metric_params, n_jobs: _minkowski_options_valid(metric, p, metric_params, n_jobs), "only dense minkowski search is covered")
@icontract.ensure(lambda result: _regressor_state_valid(result), "state must contain finite dense radius-neighbor regression data")
def radius_neighbors_regressor_fit(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    *,
    radius: float = 1.0,
    weights: str = "uniform",
    algorithm: str = "auto",
    leaf_size: int = 30,
    p: float = 2.0,
    metric: str = "minkowski",
    metric_params: None = None,
    n_jobs: None = None,
) -> NeighborsRegressorState:
    """Fit dense radius-neighbor regression state for finite numeric targets."""
    del algorithm, leaf_size, metric_params, n_jobs
    checked_x, target, outputs_2d = _checked_regression_inputs(X, y)
    return NeighborsRegressorState(
        training_data=checked_x.copy(),
        target=target.copy(),
        weights=weights,
        n_neighbors=None,
        radius=float(radius),
        metric=metric,
        p=float(p),
        regressor_kind="radius_neighbors",
        n_features_in=int(checked_x.shape[1]),
        outputs_2d=outputs_2d,
    )


@register_atom(witness_radius_neighbors_regressor_predict)
@icontract.require(lambda X: _matrix_2d(X), "X must be 2D")
@icontract.require(lambda X: _finite_matrix(X), "X must contain finite values")
@icontract.require(lambda state: _regressor_state_valid(state), "state must be a fitted dense radius-neighbor regressor")
@icontract.require(lambda state: state.regressor_kind == "radius_neighbors", "state must be a radius-neighbor regressor")
@icontract.require(lambda X, state: _regressor_feature_count_matches(X, state), "X feature count must match fitted state")
@icontract.require(lambda X, state: _radius_regressor_queries_have_neighbors(X, state), "each query must have a neighbor inside radius")
@icontract.ensure(lambda result, X, state: _regressor_prediction_valid(result, X, state), "predictions must be finite numeric targets")
def radius_neighbors_regressor_predict(X: NDArray[np.float64], state: NeighborsRegressorState) -> NDArray[np.float64]:
    """Predict finite numeric targets from all fitted samples inside a radius."""
    checked = check_array(X, dtype=np.float64, ensure_2d=True)
    distances = _pairwise_minkowski(checked, state.training_data, state.p)
    rows: list[NDArray[np.float64]] = []
    for row_distances in distances:
        neighbor_indices = np.asarray(np.flatnonzero(row_distances <= float(state.radius)), dtype=np.int64)
        if state.weights == "uniform":
            rows.append(np.asarray(np.mean(state.target[neighbor_indices], axis=0), dtype=np.float64))
        else:
            neighbor_distances = np.asarray(row_distances[neighbor_indices], dtype=np.float64)
            weights = _distance_weight_vector(neighbor_distances)
            rows.append(np.asarray(np.average(state.target[neighbor_indices], axis=0, weights=weights), dtype=np.float64))
    y_pred = np.asarray(rows, dtype=np.float64)
    if state.outputs_2d:
        return y_pred
    return np.asarray(y_pred.ravel(), dtype=np.float64)


@register_atom(witness_nearest_centroid_fit)
@icontract.require(lambda X: _matrix_2d(X), "X must be 2D")
@icontract.require(lambda y: _target_1d(y), "y must be 1D")
@icontract.require(lambda X, y: _sample_counts_match(X, y), "X and y must have matching sample counts")
@icontract.require(lambda X, y: _finite_inputs(X, y), "X and y must be finite numeric arrays")
@icontract.require(lambda y: _at_least_two_classes(y), "nearest centroid requires at least two classes")
@icontract.require(lambda X, y: _sample_count_exceeds_class_count(X, y), "sample count must exceed class count")
@icontract.require(lambda metric, shrink_threshold, priors, y: _nearest_centroid_options_valid(metric, shrink_threshold, priors, y), "nearest centroid options must be in dense numeric scope")
@icontract.require(lambda X, y, metric: _nearest_centroid_denominators_positive(X, y, metric), "class dispersion must define finite deviations")
@icontract.ensure(lambda result: _nearest_centroid_state_valid(result), "state must contain finite nearest-centroid statistics")
def nearest_centroid_fit(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    *,
    metric: str = "euclidean",
    shrink_threshold: float | None = None,
    priors: str | tuple[float, ...] = "uniform",
) -> NearestCentroidState:
    """Fit dense nearest-centroid class centroids and discriminant statistics."""
    checked_x, checked_y = check_X_y(X, y, dtype=np.float64)
    classes = np.unique(checked_y)
    n_samples, n_features = checked_x.shape
    y_ind = np.searchsorted(classes, checked_y)
    class_counts = np.bincount(y_ind, minlength=classes.shape[0]).astype(np.float64)
    if priors == "empirical":
        class_prior = class_counts / float(n_samples)
    elif priors == "uniform":
        class_prior = np.full(classes.shape[0], 1.0 / classes.shape[0], dtype=np.float64)
    else:
        raw_prior = np.asarray(priors, dtype=np.float64)
        class_prior = raw_prior / raw_prior.sum()
    centroids = _class_centroids(checked_x, checked_y, classes, metric)
    variance = (checked_x - centroids[y_ind]) ** 2
    within_std = np.asarray(np.sqrt(variance.sum(axis=0) / (n_samples - classes.shape[0])), dtype=np.float64)
    dataset_centroid = np.mean(checked_x, axis=0)
    m = np.sqrt((1.0 / class_counts) - (1.0 / n_samples))
    s = within_std + np.median(within_std)
    ms = m.reshape(classes.shape[0], 1) * s
    deviations = np.asarray((centroids - dataset_centroid) / ms, dtype=np.float64)
    fitted_centroids = centroids
    if shrink_threshold is not None:
        signs = np.sign(deviations)
        deviations = np.abs(deviations) - float(shrink_threshold)
        deviations = np.clip(deviations, 0.0, None) * signs
        fitted_centroids = np.asarray(dataset_centroid + ms * deviations, dtype=np.float64)
    return NearestCentroidState(
        classes=np.asarray(classes, dtype=np.float64),
        centroids=np.asarray(fitted_centroids, dtype=np.float64),
        deviations=np.asarray(deviations, dtype=np.float64),
        within_class_std_dev=within_std,
        class_prior=class_prior,
        metric=metric,
        shrink_threshold=None if shrink_threshold is None else float(shrink_threshold),
        n_features_in=int(n_features),
    )


@register_atom(witness_nearest_centroid_predict)
@icontract.require(lambda X: _matrix_2d(X), "X must be 2D")
@icontract.require(lambda X: _finite_matrix(X), "X must contain finite values")
@icontract.require(lambda state: _nearest_centroid_state_valid(state), "state must be a fitted nearest-centroid classifier")
@icontract.require(lambda X, state: _nearest_centroid_feature_count_matches(X, state), "X feature count must match fitted state")
@icontract.ensure(lambda result, X: _nearest_centroid_prediction_valid(result, X), "predictions must be finite class labels")
def nearest_centroid_predict(X: NDArray[np.float64], state: NearestCentroidState) -> NDArray[np.float64]:
    """Predict dense nearest-centroid class labels."""
    checked = check_array(X, dtype=np.float64, ensure_2d=True)
    if np.isclose(state.class_prior, 1.0 / state.classes.shape[0]).all():
        distances = _nearest_centroid_distances(checked, state.centroids, state.metric)
        return np.asarray(state.classes[np.argmin(distances, axis=1)], dtype=np.float64)
    scores = _nearest_centroid_raw_scores(checked, state)
    return np.asarray(state.classes[np.argmax(scores, axis=1)], dtype=np.float64)


@register_atom(witness_nearest_centroid_decision_function)
@icontract.require(lambda X: _matrix_2d(X), "X must be 2D")
@icontract.require(lambda X: _finite_matrix(X), "X must contain finite values")
@icontract.require(lambda state: _nearest_centroid_state_valid(state), "state must be a fitted nearest-centroid classifier")
@icontract.require(lambda state: state.metric == "euclidean", "decision scores are exposed for euclidean metric")
@icontract.require(lambda X, state: _nearest_centroid_feature_count_matches(X, state), "X feature count must match fitted state")
@icontract.ensure(lambda result, X, state: _nearest_centroid_scores_valid(result, X, state), "decision scores must be finite")
def nearest_centroid_decision_function(X: NDArray[np.float64], state: NearestCentroidState) -> NDArray[np.float64]:
    """Compute Euclidean nearest-centroid discriminant scores."""
    checked = check_array(X, dtype=np.float64, ensure_2d=True)
    scores = _nearest_centroid_raw_scores(checked, state)
    if state.classes.shape[0] == 2:
        return np.asarray(scores[:, 1] - scores[:, 0], dtype=np.float64)
    return scores


@register_atom(witness_nearest_centroid_predict_log_proba)
@icontract.require(lambda X: _matrix_2d(X), "X must be 2D")
@icontract.require(lambda X: _finite_matrix(X), "X must contain finite values")
@icontract.require(lambda state: _nearest_centroid_state_valid(state), "state must be a fitted nearest-centroid classifier")
@icontract.require(lambda state: state.metric == "euclidean", "log probabilities are exposed for euclidean metric")
@icontract.require(lambda X, state: _nearest_centroid_feature_count_matches(X, state), "X feature count must match fitted state")
@icontract.ensure(lambda result, X, state: _nearest_centroid_log_proba_valid(result, X, state), "log probabilities must normalize")
def nearest_centroid_predict_log_proba(X: NDArray[np.float64], state: NearestCentroidState) -> NDArray[np.float64]:
    """Compute Euclidean nearest-centroid log class probabilities."""
    checked = check_array(X, dtype=np.float64, ensure_2d=True)
    scores = _nearest_centroid_raw_scores(checked, state)
    shifted = scores - scores.max(axis=1)[:, np.newaxis]
    return np.asarray(shifted - np.log(np.exp(shifted).sum(axis=1)[:, np.newaxis]), dtype=np.float64)


@register_atom(witness_nearest_centroid_predict_proba)
@icontract.require(lambda X: _matrix_2d(X), "X must be 2D")
@icontract.require(lambda X: _finite_matrix(X), "X must contain finite values")
@icontract.require(lambda state: _nearest_centroid_state_valid(state), "state must be a fitted nearest-centroid classifier")
@icontract.require(lambda state: state.metric == "euclidean", "probabilities are exposed for euclidean metric")
@icontract.require(lambda X, state: _nearest_centroid_feature_count_matches(X, state), "X feature count must match fitted state")
@icontract.ensure(lambda result, X, state: _nearest_centroid_proba_valid(result, X, state), "probabilities must normalize")
def nearest_centroid_predict_proba(X: NDArray[np.float64], state: NearestCentroidState) -> NDArray[np.float64]:
    """Compute Euclidean nearest-centroid class probabilities."""
    return _softmax(_nearest_centroid_raw_scores(check_array(X, dtype=np.float64, ensure_2d=True), state))
