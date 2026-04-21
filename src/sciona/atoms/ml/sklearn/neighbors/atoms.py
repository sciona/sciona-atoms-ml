"""Dense neighbors graph atoms adapted from scikit-learn."""

from __future__ import annotations

import math

import icontract
import numpy as np
from numpy.typing import NDArray
from scipy.special import gammainc
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_X_y, check_array

from sciona.ghost.registry import register_atom

from .state_models import (
    KernelDensityState,
    NearestCentroidState,
    NearestNeighborsState,
    NeighborsClassifierState,
    NeighborsGraphTransformerState,
    NeighborsRegressorState,
)
from .witnesses import (
    witness_kneighbors_classifier_fit,
    witness_kneighbors_classifier_predict,
    witness_kneighbors_classifier_predict_proba,
    witness_kneighbors_graph,
    witness_kneighbors_regressor_fit,
    witness_kneighbors_regressor_predict,
    witness_kneighbors_transform,
    witness_kneighbors_transformer_fit,
    witness_kernel_density_fit,
    witness_kernel_density_sample,
    witness_kernel_density_score,
    witness_kernel_density_score_samples,
    witness_nearest_centroid_decision_function,
    witness_nearest_centroid_fit,
    witness_nearest_centroid_predict,
    witness_nearest_centroid_predict_log_proba,
    witness_nearest_centroid_predict_proba,
    witness_nearest_neighbors_fit,
    witness_nearest_neighbors_kneighbors,
    witness_nearest_neighbors_kneighbors_graph,
    witness_nearest_neighbors_radius_neighbors,
    witness_nearest_neighbors_radius_neighbors_graph,
    witness_radius_neighbors_classifier_fit,
    witness_radius_neighbors_classifier_predict,
    witness_radius_neighbors_classifier_predict_proba,
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


def _finite_classification_inputs(X: NDArray[np.float64], y: NDArray[np.float64]) -> bool:
    try:
        values_x = np.asarray(X, dtype=np.float64)
        values_y = np.asarray(y, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(values_y.ndim == 1 and np.all(np.isfinite(values_x)) and np.all(np.isfinite(values_y)))


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


def _classifier_state_valid(state: NeighborsClassifierState) -> bool:
    n_samples = state.training_data.shape[0]
    return bool(
        state.training_data.ndim == 2
        and state.training_data.shape[1] == state.n_features_in
        and state.labels.ndim == 1
        and state.labels.shape[0] == n_samples
        and state.classes.ndim == 1
        and state.classes.shape[0] >= 2
        and state.weights in {"uniform", "distance"}
        and state.metric == "minkowski"
        and np.isfinite(state.p)
        and state.p >= 1.0
        and state.classifier_kind in {"kneighbors", "radius_neighbors"}
        and np.all(np.isfinite(state.training_data))
        and np.all(np.isfinite(state.classes))
        and np.all(state.labels >= 0)
        and np.all(state.labels < state.classes.shape[0])
        and (
            (
                state.classifier_kind == "kneighbors"
                and state.n_neighbors is not None
                and 1 <= state.n_neighbors <= n_samples
                and state.radius is None
            )
            or (
                state.classifier_kind == "radius_neighbors"
                and state.n_neighbors is None
                and state.radius is not None
                and np.isfinite(state.radius)
                and state.radius >= 0.0
            )
        )
    )


def _classifier_feature_count_matches(X: NDArray[np.float64], state: NeighborsClassifierState) -> bool:
    values = np.asarray(X)
    return bool(values.ndim == 2 and values.shape[1] == state.n_features_in)


def _classifier_prediction_valid(result: NDArray[np.float64], X: NDArray[np.float64], state: NeighborsClassifierState) -> bool:
    values = np.asarray(result, dtype=np.float64)
    return bool(
        values.shape == (np.asarray(X).shape[0],)
        and np.all(np.isfinite(values))
        and np.all(np.isin(values, state.classes))
    )


def _classifier_proba_valid(result: NDArray[np.float64], X: NDArray[np.float64], state: NeighborsClassifierState) -> bool:
    values = np.asarray(result, dtype=np.float64)
    return bool(
        values.shape == (np.asarray(X).shape[0], state.classes.shape[0])
        and np.all(np.isfinite(values))
        and np.all(values >= 0.0)
        and np.all(values <= 1.0)
        and np.allclose(np.sum(values, axis=1), 1.0)
    )


def _radius_classifier_queries_have_neighbors(X: NDArray[np.float64], state: NeighborsClassifierState) -> bool:
    try:
        values = np.asarray(X, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    if not _classifier_state_valid(state) or state.classifier_kind != "radius_neighbors" or values.ndim != 2:
        return False
    if values.shape[1] != state.n_features_in:
        return False
    distances = _pairwise_minkowski(values, state.training_data, state.p)
    return bool(np.all(np.any(distances <= float(state.radius), axis=1)))


def _nearest_neighbors_state_valid(state: NearestNeighborsState) -> bool:
    n_samples = state.training_data.shape[0]
    return bool(
        state.training_data.ndim == 2
        and state.training_data.shape[1] == state.n_features_in
        and n_samples >= 1
        and 1 <= state.n_neighbors <= n_samples
        and np.isfinite(state.radius)
        and state.radius >= 0.0
        and state.metric == "minkowski"
        and np.isfinite(state.p)
        and state.p >= 1.0
        and np.all(np.isfinite(state.training_data))
    )


def _nearest_neighbors_feature_count_matches(X: NDArray[np.float64], state: NearestNeighborsState) -> bool:
    values = np.asarray(X)
    return bool(values.ndim == 2 and values.shape[1] == state.n_features_in)


def _n_neighbors_query_valid(n_neighbors: int | None, state: NearestNeighborsState) -> bool:
    if not _nearest_neighbors_state_valid(state):
        return False
    if n_neighbors is None:
        return True
    return bool(
        isinstance(n_neighbors, int)
        and not isinstance(n_neighbors, bool)
        and 1 <= n_neighbors <= state.training_data.shape[0]
    )


def _optional_radius_valid(radius: float | None) -> bool:
    return radius is None or _radius_valid(radius)


def _kneighbors_query_result_valid(
    result: tuple[NDArray[np.float64], NDArray[np.int64]],
    X: NDArray[np.float64],
    state: NearestNeighborsState,
    n_neighbors: int | None,
) -> bool:
    distances, indices = result
    k = state.n_neighbors if n_neighbors is None else int(n_neighbors)
    return bool(
        distances.shape == (np.asarray(X).shape[0], k)
        and indices.shape == (np.asarray(X).shape[0], k)
        and np.all(np.isfinite(distances))
        and np.all(distances >= 0.0)
        and np.all(indices >= 0)
        and np.all(indices < state.training_data.shape[0])
    )


def _object_neighbor_rows_valid(
    rows: NDArray[np.object_],
    n_queries: int,
    state: NearestNeighborsState,
    *,
    distance_rows: bool,
) -> bool:
    if rows.shape != (n_queries,):
        return False
    for row in rows:
        values = np.asarray(row, dtype=np.float64 if distance_rows else np.int64)
        if values.ndim != 1:
            return False
        if distance_rows and (not np.all(np.isfinite(values)) or not np.all(values >= 0.0)):
            return False
        if not distance_rows and (not np.all(values >= 0) or not np.all(values < state.training_data.shape[0])):
            return False
    return True


def _radius_neighbors_query_result_valid(
    result: tuple[NDArray[np.object_], NDArray[np.object_]],
    X: NDArray[np.float64],
    state: NearestNeighborsState,
) -> bool:
    distances, indices = result
    n_queries = int(np.asarray(X).shape[0])
    return bool(
        _object_neighbor_rows_valid(distances, n_queries, state, distance_rows=True)
        and _object_neighbor_rows_valid(indices, n_queries, state, distance_rows=False)
        and all(np.asarray(d).shape == np.asarray(i).shape for d, i in zip(distances, indices))
    )


def _kernel_density_kernel_valid(kernel: str) -> bool:
    return kernel in {"gaussian", "tophat", "epanechnikov", "exponential", "linear", "cosine"}


def _kernel_density_bandwidth_valid(bandwidth: float | str) -> bool:
    if bandwidth in {"scott", "silverman"}:
        return True
    return bool(
        isinstance(bandwidth, (int, float))
        and not isinstance(bandwidth, bool)
        and np.isfinite(float(bandwidth))
        and float(bandwidth) > 0.0
    )


def _kernel_density_options_valid(
    algorithm: str,
    kernel: str,
    metric: str,
    atol: float,
    rtol: float,
    breadth_first: bool,
    leaf_size: int,
    metric_params: None,
) -> bool:
    return bool(
        algorithm in {"auto", "kd_tree", "ball_tree"}
        and _kernel_density_kernel_valid(kernel)
        and metric == "euclidean"
        and isinstance(atol, (int, float))
        and not isinstance(atol, bool)
        and np.isfinite(float(atol))
        and float(atol) >= 0.0
        and isinstance(rtol, (int, float))
        and not isinstance(rtol, bool)
        and np.isfinite(float(rtol))
        and float(rtol) >= 0.0
        and isinstance(breadth_first, bool)
        and isinstance(leaf_size, int)
        and not isinstance(leaf_size, bool)
        and leaf_size >= 1
        and metric_params is None
    )


def _sample_weight_valid(sample_weight: NDArray[np.float64] | None, X: NDArray[np.float64]) -> bool:
    if sample_weight is None:
        return True
    try:
        weights = np.asarray(sample_weight, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(
        weights.ndim == 1
        and weights.shape[0] == np.asarray(X).shape[0]
        and np.all(np.isfinite(weights))
        and np.all(weights >= 0.0)
        and np.sum(weights) > 0.0
    )


def _kernel_density_state_valid(state: KernelDensityState) -> bool:
    n_samples = state.training_data.shape[0]
    weights_valid = (
        state.sample_weight is None
        or (
            state.sample_weight.ndim == 1
            and state.sample_weight.shape[0] == n_samples
            and np.all(np.isfinite(state.sample_weight))
            and np.all(state.sample_weight >= 0.0)
            and np.sum(state.sample_weight) > 0.0
        )
    )
    return bool(
        state.training_data.ndim == 2
        and n_samples >= 1
        and state.training_data.shape[1] == state.n_features_in
        and np.all(np.isfinite(state.training_data))
        and np.isfinite(state.bandwidth)
        and state.bandwidth > 0.0
        and _kernel_density_kernel_valid(state.kernel)
        and state.metric == "euclidean"
        and np.isfinite(state.atol)
        and state.atol >= 0.0
        and np.isfinite(state.rtol)
        and state.rtol >= 0.0
        and isinstance(state.breadth_first, bool)
        and weights_valid
    )


def _kernel_density_feature_count_matches(X: NDArray[np.float64], state: KernelDensityState) -> bool:
    values = np.asarray(X)
    return bool(values.ndim == 2 and values.shape[1] == state.n_features_in)


def _log_density_valid(result: NDArray[np.float64], X: NDArray[np.float64]) -> bool:
    values = np.asarray(result, dtype=np.float64)
    return bool(values.shape == (np.asarray(X).shape[0],) and not np.any(np.isnan(values)))


def _log_score_valid(result: float) -> bool:
    return bool(isinstance(result, (int, float, np.floating)) and not np.isnan(float(result)))


def _positive_sample_count(n_samples: int) -> bool:
    return bool(isinstance(n_samples, int) and not isinstance(n_samples, bool) and n_samples >= 1)


def _kernel_density_sampling_state_valid(state: KernelDensityState) -> bool:
    return bool(_kernel_density_state_valid(state) and state.kernel in {"gaussian", "tophat"})


def _kernel_density_sample_valid(result: NDArray[np.float64], state: KernelDensityState, n_samples: int) -> bool:
    values = np.asarray(result, dtype=np.float64)
    return bool(values.shape == (n_samples, state.n_features_in) and np.all(np.isfinite(values)))


def _object_array_from_rows(rows: list[NDArray[np.float64]] | list[NDArray[np.int64]]) -> NDArray[np.object_]:
    result = np.empty(len(rows), dtype=object)
    for row_index, row in enumerate(rows):
        result[row_index] = row
    return result


def _kernel_density_bandwidth(bandwidth: float | str, n_samples: int, n_features: int) -> float:
    if bandwidth == "scott":
        return float(n_samples ** (-1.0 / (n_features + 4.0)))
    if bandwidth == "silverman":
        return float((n_samples * (n_features + 2.0) / 4.0) ** (-1.0 / (n_features + 4.0)))
    return float(bandwidth)


def _log_unit_ball_volume(dimension: int) -> float:
    return 0.5 * dimension * math.log(math.pi) - math.lgamma(0.5 * dimension + 1.0)


def _log_sphere_surface(dimension: int) -> float:
    return math.log(2.0 * math.pi) + _log_unit_ball_volume(dimension - 1)


def _kernel_log_norm(bandwidth: float, dimension: int, kernel: str) -> float:
    if kernel == "gaussian":
        factor = 0.5 * dimension * math.log(2.0 * math.pi)
    elif kernel == "tophat":
        factor = _log_unit_ball_volume(dimension)
    elif kernel == "epanechnikov":
        factor = _log_unit_ball_volume(dimension) + math.log(2.0 / (dimension + 2.0))
    elif kernel == "exponential":
        factor = _log_sphere_surface(dimension - 1) + math.lgamma(dimension)
    elif kernel == "linear":
        factor = _log_unit_ball_volume(dimension) - math.log(dimension + 1.0)
    else:
        factor_value = 0.0
        term = 2.0 / math.pi
        for k in range(1, dimension + 1, 2):
            factor_value += term
            term *= -float((dimension - k) * (dimension - k - 1)) * (2.0 / math.pi) ** 2
        factor = math.log(factor_value) + _log_sphere_surface(dimension - 1)
    return -factor - dimension * math.log(bandwidth)


def _kernel_log_values(distances: NDArray[np.float64], bandwidth: float, kernel: str) -> NDArray[np.float64]:
    scaled = np.asarray(distances, dtype=np.float64) / float(bandwidth)
    if kernel == "gaussian":
        return np.asarray(-0.5 * scaled * scaled, dtype=np.float64)
    if kernel == "exponential":
        return np.asarray(-scaled, dtype=np.float64)
    log_values = np.full(scaled.shape, -np.inf, dtype=np.float64)
    inside = scaled < 1.0
    if kernel == "tophat":
        log_values[inside] = 0.0
    elif kernel == "epanechnikov":
        log_values[inside] = np.log1p(-(scaled[inside] * scaled[inside]))
    elif kernel == "linear":
        log_values[inside] = np.log1p(-scaled[inside])
    else:
        log_values[inside] = np.log(np.cos(0.5 * math.pi * scaled[inside]))
    return log_values


def _logsumexp_rows(values: NDArray[np.float64]) -> NDArray[np.float64]:
    row_max = np.max(values, axis=1)
    result = np.full(row_max.shape, -np.inf, dtype=np.float64)
    finite = np.isfinite(row_max)
    if np.any(finite):
        shifted = values[finite] - row_max[finite, np.newaxis]
        result[finite] = row_max[finite] + np.log(np.sum(np.exp(shifted), axis=1))
    return result


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
    state: NeighborsRegressorState | NeighborsClassifierState | NearestNeighborsState,
    n_neighbors: int | None = None,
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    k = int(state.n_neighbors if n_neighbors is None else n_neighbors)
    distances = _pairwise_minkowski(X, state.training_data, state.p)
    order = np.argsort(distances, axis=1, kind="stable")[:, :k]
    rows = np.arange(distances.shape[0])[:, np.newaxis]
    return np.asarray(order, dtype=np.int64), np.asarray(distances[rows, order], dtype=np.float64)


def _checked_classification_inputs(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.float64]]:
    checked_x, checked_y = check_X_y(X, y, dtype=np.float64)
    labels_raw = np.asarray(checked_y, dtype=np.float64)
    classes = np.unique(labels_raw)
    encoded = np.searchsorted(classes, labels_raw)
    return np.asarray(checked_x, dtype=np.float64), np.asarray(encoded, dtype=np.int64), np.asarray(classes, dtype=np.float64)


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


def _class_probabilities(
    labels: NDArray[np.int64],
    classes: NDArray[np.float64],
    neighbor_indices: NDArray[np.int64],
    weights: NDArray[np.float64],
) -> NDArray[np.float64]:
    probabilities = np.zeros((neighbor_indices.shape[0], classes.shape[0]), dtype=np.float64)
    rows = np.arange(neighbor_indices.shape[0])
    for neighbor_position in range(neighbor_indices.shape[1]):
        probabilities[rows, labels[neighbor_indices[:, neighbor_position]]] += weights[:, neighbor_position]
    normalizer = probabilities.sum(axis=1)[:, np.newaxis]
    return np.asarray(probabilities / normalizer, dtype=np.float64)


def _radius_class_probabilities(
    labels: NDArray[np.int64],
    classes: NDArray[np.float64],
    distances: NDArray[np.float64],
    radius: float,
    weights_kind: str,
) -> NDArray[np.float64]:
    probabilities = np.zeros((distances.shape[0], classes.shape[0]), dtype=np.float64)
    for row_index, row_distances in enumerate(distances):
        neighbor_indices = np.asarray(np.flatnonzero(row_distances <= float(radius)), dtype=np.int64)
        if weights_kind == "uniform":
            weights = np.ones(neighbor_indices.shape[0], dtype=np.float64)
        else:
            weights = _distance_weight_vector(np.asarray(row_distances[neighbor_indices], dtype=np.float64))
        probabilities[row_index] = np.bincount(labels[neighbor_indices], weights=weights, minlength=classes.shape[0])
    normalizer = probabilities.sum(axis=1)[:, np.newaxis]
    return np.asarray(probabilities / normalizer, dtype=np.float64)


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


@register_atom(witness_kneighbors_classifier_fit)
@icontract.require(lambda X: _matrix_2d(X), "X must be 2D")
@icontract.require(lambda y: _target_1d(y), "y must be 1D")
@icontract.require(lambda X, y: _sample_counts_match(X, y), "X and y must have matching sample counts")
@icontract.require(lambda X, y: _finite_classification_inputs(X, y), "X and labels must be finite numeric arrays")
@icontract.require(lambda y: _at_least_two_classes(y), "classifier requires at least two classes")
@icontract.require(lambda n_neighbors, X: _positive_neighbors(n_neighbors, X), "n_neighbors must fit sample count")
@icontract.require(lambda weights: _weights_valid(weights), "weights must be uniform or distance")
@icontract.require(lambda algorithm, leaf_size: _algorithm_options_valid(algorithm, leaf_size), "algorithm and leaf_size must be valid")
@icontract.require(lambda metric, p, metric_params, n_jobs: _minkowski_options_valid(metric, p, metric_params, n_jobs), "only dense minkowski search is covered")
@icontract.ensure(lambda result: _classifier_state_valid(result), "state must contain finite dense k-neighbor classification data")
def kneighbors_classifier_fit(
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
) -> NeighborsClassifierState:
    """Fit dense k-neighbor classification state for finite numeric labels."""
    del algorithm, leaf_size, metric_params, n_jobs
    checked_x, labels, classes = _checked_classification_inputs(X, y)
    return NeighborsClassifierState(
        training_data=checked_x.copy(),
        labels=labels.copy(),
        classes=classes.copy(),
        weights=weights,
        n_neighbors=int(n_neighbors),
        radius=None,
        metric=metric,
        p=float(p),
        classifier_kind="kneighbors",
        n_features_in=int(checked_x.shape[1]),
    )


@register_atom(witness_kneighbors_classifier_predict_proba)
@icontract.require(lambda X: _matrix_2d(X), "X must be 2D")
@icontract.require(lambda X: _finite_matrix(X), "X must contain finite values")
@icontract.require(lambda state: _classifier_state_valid(state), "state must be a fitted dense k-neighbor classifier")
@icontract.require(lambda state: state.classifier_kind == "kneighbors", "state must be a k-neighbor classifier")
@icontract.require(lambda X, state: _classifier_feature_count_matches(X, state), "X feature count must match fitted state")
@icontract.ensure(lambda result, X, state: _classifier_proba_valid(result, X, state), "probabilities must normalize")
def kneighbors_classifier_predict_proba(X: NDArray[np.float64], state: NeighborsClassifierState) -> NDArray[np.float64]:
    """Compute dense k-neighbor class probabilities for numeric labels."""
    checked = check_array(X, dtype=np.float64, ensure_2d=True)
    neighbor_indices, neighbor_distances = _kneighbor_indices_and_distances(checked, state)
    if state.weights == "uniform":
        weights = np.ones_like(neighbor_distances, dtype=np.float64)
    else:
        weights = _distance_weight_matrix(neighbor_distances)
    return _class_probabilities(state.labels, state.classes, neighbor_indices, weights)


@register_atom(witness_kneighbors_classifier_predict)
@icontract.require(lambda X: _matrix_2d(X), "X must be 2D")
@icontract.require(lambda X: _finite_matrix(X), "X must contain finite values")
@icontract.require(lambda state: _classifier_state_valid(state), "state must be a fitted dense k-neighbor classifier")
@icontract.require(lambda state: state.classifier_kind == "kneighbors", "state must be a k-neighbor classifier")
@icontract.require(lambda X, state: _classifier_feature_count_matches(X, state), "X feature count must match fitted state")
@icontract.ensure(lambda result, X, state: _classifier_prediction_valid(result, X, state), "predictions must be finite class labels")
def kneighbors_classifier_predict(X: NDArray[np.float64], state: NeighborsClassifierState) -> NDArray[np.float64]:
    """Predict numeric class labels by dense k-neighbor vote."""
    probabilities = kneighbors_classifier_predict_proba(X, state)
    return np.asarray(state.classes[np.argmax(probabilities, axis=1)], dtype=np.float64)


@register_atom(witness_radius_neighbors_classifier_fit)
@icontract.require(lambda X: _matrix_2d(X), "X must be 2D")
@icontract.require(lambda y: _target_1d(y), "y must be 1D")
@icontract.require(lambda X, y: _sample_counts_match(X, y), "X and y must have matching sample counts")
@icontract.require(lambda X, y: _finite_classification_inputs(X, y), "X and labels must be finite numeric arrays")
@icontract.require(lambda y: _at_least_two_classes(y), "classifier requires at least two classes")
@icontract.require(lambda radius: _radius_valid(radius), "radius must be nonnegative and finite")
@icontract.require(lambda weights: _weights_valid(weights), "weights must be uniform or distance")
@icontract.require(lambda outlier_label: outlier_label is None, "outlier labels are outside this atom scope")
@icontract.require(lambda algorithm, leaf_size: _algorithm_options_valid(algorithm, leaf_size), "algorithm and leaf_size must be valid")
@icontract.require(lambda metric, p, metric_params, n_jobs: _minkowski_options_valid(metric, p, metric_params, n_jobs), "only dense minkowski search is covered")
@icontract.ensure(lambda result: _classifier_state_valid(result), "state must contain finite dense radius-neighbor classification data")
def radius_neighbors_classifier_fit(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    *,
    radius: float = 1.0,
    weights: str = "uniform",
    algorithm: str = "auto",
    leaf_size: int = 30,
    p: float = 2.0,
    metric: str = "minkowski",
    outlier_label: None = None,
    metric_params: None = None,
    n_jobs: None = None,
) -> NeighborsClassifierState:
    """Fit dense radius-neighbor classification state for finite numeric labels."""
    del algorithm, leaf_size, outlier_label, metric_params, n_jobs
    checked_x, labels, classes = _checked_classification_inputs(X, y)
    return NeighborsClassifierState(
        training_data=checked_x.copy(),
        labels=labels.copy(),
        classes=classes.copy(),
        weights=weights,
        n_neighbors=None,
        radius=float(radius),
        metric=metric,
        p=float(p),
        classifier_kind="radius_neighbors",
        n_features_in=int(checked_x.shape[1]),
    )


@register_atom(witness_radius_neighbors_classifier_predict_proba)
@icontract.require(lambda X: _matrix_2d(X), "X must be 2D")
@icontract.require(lambda X: _finite_matrix(X), "X must contain finite values")
@icontract.require(lambda state: _classifier_state_valid(state), "state must be a fitted dense radius-neighbor classifier")
@icontract.require(lambda state: state.classifier_kind == "radius_neighbors", "state must be a radius-neighbor classifier")
@icontract.require(lambda X, state: _classifier_feature_count_matches(X, state), "X feature count must match fitted state")
@icontract.require(lambda X, state: _radius_classifier_queries_have_neighbors(X, state), "each query must have a neighbor inside radius")
@icontract.ensure(lambda result, X, state: _classifier_proba_valid(result, X, state), "probabilities must normalize")
def radius_neighbors_classifier_predict_proba(X: NDArray[np.float64], state: NeighborsClassifierState) -> NDArray[np.float64]:
    """Compute dense radius-neighbor class probabilities for numeric labels."""
    checked = check_array(X, dtype=np.float64, ensure_2d=True)
    distances = _pairwise_minkowski(checked, state.training_data, state.p)
    return _radius_class_probabilities(state.labels, state.classes, distances, float(state.radius), state.weights)


@register_atom(witness_radius_neighbors_classifier_predict)
@icontract.require(lambda X: _matrix_2d(X), "X must be 2D")
@icontract.require(lambda X: _finite_matrix(X), "X must contain finite values")
@icontract.require(lambda state: _classifier_state_valid(state), "state must be a fitted dense radius-neighbor classifier")
@icontract.require(lambda state: state.classifier_kind == "radius_neighbors", "state must be a radius-neighbor classifier")
@icontract.require(lambda X, state: _classifier_feature_count_matches(X, state), "X feature count must match fitted state")
@icontract.require(lambda X, state: _radius_classifier_queries_have_neighbors(X, state), "each query must have a neighbor inside radius")
@icontract.ensure(lambda result, X, state: _classifier_prediction_valid(result, X, state), "predictions must be finite class labels")
def radius_neighbors_classifier_predict(X: NDArray[np.float64], state: NeighborsClassifierState) -> NDArray[np.float64]:
    """Predict numeric class labels by dense radius-neighbor vote."""
    probabilities = radius_neighbors_classifier_predict_proba(X, state)
    return np.asarray(state.classes[np.argmax(probabilities, axis=1)], dtype=np.float64)


@register_atom(witness_nearest_neighbors_fit)
@icontract.require(lambda X: _matrix_2d(X), "X must be 2D")
@icontract.require(lambda X: _finite_matrix(X), "X must contain finite values")
@icontract.require(lambda n_neighbors, X: _positive_neighbors(n_neighbors, X), "n_neighbors must fit sample count")
@icontract.require(lambda radius: _radius_valid(radius), "radius must be nonnegative and finite")
@icontract.require(lambda algorithm, leaf_size: _algorithm_options_valid(algorithm, leaf_size), "algorithm and leaf_size must be valid")
@icontract.require(lambda metric, p, metric_params, n_jobs: _minkowski_options_valid(metric, p, metric_params, n_jobs), "only dense minkowski search is covered")
@icontract.ensure(lambda result: _nearest_neighbors_state_valid(result), "state must contain finite dense nearest-neighbor data")
def nearest_neighbors_fit(
    X: NDArray[np.float64],
    *,
    n_neighbors: int = 5,
    radius: float = 1.0,
    algorithm: str = "auto",
    leaf_size: int = 30,
    metric: str = "minkowski",
    p: float = 2.0,
    metric_params: None = None,
    n_jobs: None = None,
) -> NearestNeighborsState:
    """Fit dense nearest-neighbor search state for finite numeric samples."""
    del algorithm, leaf_size, metric_params, n_jobs
    checked = check_array(X, dtype=np.float64, ensure_2d=True)
    return NearestNeighborsState(
        training_data=np.asarray(checked, dtype=np.float64).copy(),
        n_neighbors=int(n_neighbors),
        radius=float(radius),
        metric=metric,
        p=float(p),
        n_features_in=int(checked.shape[1]),
    )


@register_atom(witness_nearest_neighbors_kneighbors)
@icontract.require(lambda X: _matrix_2d(X), "X must be 2D")
@icontract.require(lambda X: _finite_matrix(X), "X must contain finite values")
@icontract.require(lambda state: _nearest_neighbors_state_valid(state), "state must be a fitted dense nearest-neighbor search")
@icontract.require(lambda X, state: _nearest_neighbors_feature_count_matches(X, state), "X feature count must match fitted state")
@icontract.require(lambda n_neighbors, state: _n_neighbors_query_valid(n_neighbors, state), "n_neighbors must fit fitted sample count")
@icontract.ensure(lambda result, X, state, n_neighbors: _kneighbors_query_result_valid(result, X, state, n_neighbors), "k-neighbor query must return finite distances and valid indices")
def nearest_neighbors_kneighbors(
    X: NDArray[np.float64],
    state: NearestNeighborsState,
    n_neighbors: int | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Return dense k-neighbor distances and fitted-row indices for queries."""
    checked = check_array(X, dtype=np.float64, ensure_2d=True)
    indices, distances = _kneighbor_indices_and_distances(checked, state, n_neighbors)
    return distances, indices


@register_atom(witness_nearest_neighbors_radius_neighbors)
@icontract.require(lambda X: _matrix_2d(X), "X must be 2D")
@icontract.require(lambda X: _finite_matrix(X), "X must contain finite values")
@icontract.require(lambda state: _nearest_neighbors_state_valid(state), "state must be a fitted dense nearest-neighbor search")
@icontract.require(lambda X, state: _nearest_neighbors_feature_count_matches(X, state), "X feature count must match fitted state")
@icontract.require(lambda radius: _optional_radius_valid(radius), "radius must be nonnegative and finite when provided")
@icontract.ensure(lambda result, X, state: _radius_neighbors_query_result_valid(result, X, state), "radius query must return finite distances and valid indices")
def nearest_neighbors_radius_neighbors(
    X: NDArray[np.float64],
    state: NearestNeighborsState,
    radius: float | None = None,
    *,
    sort_results: bool = False,
) -> tuple[NDArray[np.object_], NDArray[np.object_]]:
    """Return ragged radius-neighbor distances and fitted-row indices."""
    checked = check_array(X, dtype=np.float64, ensure_2d=True)
    radius_value = state.radius if radius is None else float(radius)
    distances = _pairwise_minkowski(checked, state.training_data, state.p)
    distance_rows: list[NDArray[np.float64]] = []
    index_rows: list[NDArray[np.int64]] = []
    for row_distances in distances:
        indices = np.asarray(np.flatnonzero(row_distances <= radius_value), dtype=np.int64)
        selected = np.asarray(row_distances[indices], dtype=np.float64)
        if sort_results:
            order = np.argsort(selected, kind="stable")
            indices = indices[order]
            selected = selected[order]
        distance_rows.append(selected)
        index_rows.append(indices)
    return _object_array_from_rows(distance_rows), _object_array_from_rows(index_rows)


@register_atom(witness_nearest_neighbors_kneighbors_graph)
@icontract.require(lambda X: _matrix_2d(X), "X must be 2D")
@icontract.require(lambda X: _finite_matrix(X), "X must contain finite values")
@icontract.require(lambda state: _nearest_neighbors_state_valid(state), "state must be a fitted dense nearest-neighbor search")
@icontract.require(lambda X, state: _nearest_neighbors_feature_count_matches(X, state), "X feature count must match fitted state")
@icontract.require(lambda n_neighbors, state: _n_neighbors_query_valid(n_neighbors, state), "n_neighbors must fit fitted sample count")
@icontract.require(lambda mode: _mode_valid(mode), "mode must be connectivity or distance")
@icontract.ensure(lambda result, X, state: _graph_valid(result, np.asarray(X).shape[0], state.training_data.shape[0]), "graph must be finite and nonnegative")
def nearest_neighbors_kneighbors_graph(
    X: NDArray[np.float64],
    state: NearestNeighborsState,
    n_neighbors: int | None = None,
    *,
    mode: str = "connectivity",
) -> NDArray[np.float64]:
    """Build a dense fitted k-neighbor connectivity or distance graph."""
    distances, indices = nearest_neighbors_kneighbors(X, state, n_neighbors)
    graph = np.zeros((distances.shape[0], state.training_data.shape[0]), dtype=np.float64)
    rows = np.arange(distances.shape[0])[:, np.newaxis]
    if mode == "connectivity":
        graph[rows, indices] = 1.0
    else:
        graph[rows, indices] = distances
    return graph


@register_atom(witness_nearest_neighbors_radius_neighbors_graph)
@icontract.require(lambda X: _matrix_2d(X), "X must be 2D")
@icontract.require(lambda X: _finite_matrix(X), "X must contain finite values")
@icontract.require(lambda state: _nearest_neighbors_state_valid(state), "state must be a fitted dense nearest-neighbor search")
@icontract.require(lambda X, state: _nearest_neighbors_feature_count_matches(X, state), "X feature count must match fitted state")
@icontract.require(lambda radius: _optional_radius_valid(radius), "radius must be nonnegative and finite when provided")
@icontract.require(lambda mode: _mode_valid(mode), "mode must be connectivity or distance")
@icontract.ensure(lambda result, X, state: _graph_valid(result, np.asarray(X).shape[0], state.training_data.shape[0]), "graph must be finite and nonnegative")
def nearest_neighbors_radius_neighbors_graph(
    X: NDArray[np.float64],
    state: NearestNeighborsState,
    radius: float | None = None,
    *,
    mode: str = "connectivity",
    sort_results: bool = False,
) -> NDArray[np.float64]:
    """Build a dense fitted radius-neighbor connectivity or distance graph."""
    distances, indices = nearest_neighbors_radius_neighbors(X, state, radius, sort_results=sort_results)
    graph = np.zeros((distances.shape[0], state.training_data.shape[0]), dtype=np.float64)
    for row_index, row_indices in enumerate(indices):
        row_indices_int = np.asarray(row_indices, dtype=np.int64)
        if mode == "connectivity":
            graph[row_index, row_indices_int] = 1.0
        else:
            graph[row_index, row_indices_int] = np.asarray(distances[row_index], dtype=np.float64)
    return graph


@register_atom(witness_kernel_density_fit)
@icontract.require(lambda X: _matrix_2d(X), "X must be 2D")
@icontract.require(lambda X: _finite_matrix(X), "X must contain finite values")
@icontract.require(lambda bandwidth: _kernel_density_bandwidth_valid(bandwidth), "bandwidth must be positive or a supported rule")
@icontract.require(lambda sample_weight, X: _sample_weight_valid(sample_weight, X), "sample weights must be finite nonnegative values")
@icontract.require(lambda algorithm, kernel, metric, atol, rtol, breadth_first, leaf_size, metric_params: _kernel_density_options_valid(algorithm, kernel, metric, atol, rtol, breadth_first, leaf_size, metric_params), "kernel density options must be in dense Euclidean scope")
@icontract.ensure(lambda result: _kernel_density_state_valid(result), "state must contain finite dense kernel-density data")
def kernel_density_fit(
    X: NDArray[np.float64],
    *,
    bandwidth: float | str = 1.0,
    algorithm: str = "auto",
    kernel: str = "gaussian",
    metric: str = "euclidean",
    atol: float = 0.0,
    rtol: float = 0.0,
    breadth_first: bool = True,
    leaf_size: int = 40,
    metric_params: None = None,
    sample_weight: NDArray[np.float64] | None = None,
) -> KernelDensityState:
    """Fit dense Euclidean kernel-density state for finite numeric samples."""
    del algorithm, leaf_size, metric_params
    checked = check_array(X, dtype=np.float64, ensure_2d=True)
    weights = None if sample_weight is None else np.asarray(sample_weight, dtype=np.float64).copy()
    return KernelDensityState(
        training_data=np.asarray(checked, dtype=np.float64).copy(),
        sample_weight=weights,
        bandwidth=_kernel_density_bandwidth(bandwidth, int(checked.shape[0]), int(checked.shape[1])),
        kernel=kernel,
        metric=metric,
        atol=float(atol),
        rtol=float(rtol),
        breadth_first=breadth_first,
        n_features_in=int(checked.shape[1]),
    )


@register_atom(witness_kernel_density_score_samples)
@icontract.require(lambda X: _matrix_2d(X), "X must be 2D")
@icontract.require(lambda X: _finite_matrix(X), "X must contain finite values")
@icontract.require(lambda state: _kernel_density_state_valid(state), "state must be a fitted dense kernel-density estimator")
@icontract.require(lambda X, state: _kernel_density_feature_count_matches(X, state), "X feature count must match fitted state")
@icontract.ensure(lambda result, X: _log_density_valid(result, X), "log densities must be shaped and non-NaN")
def kernel_density_score_samples(X: NDArray[np.float64], state: KernelDensityState) -> NDArray[np.float64]:
    """Compute per-sample Euclidean kernel log densities."""
    checked = check_array(X, dtype=np.float64, ensure_2d=True)
    distances = _pairwise_minkowski(checked, state.training_data, 2.0)
    log_terms = _kernel_log_values(distances, state.bandwidth, state.kernel)
    log_terms += _kernel_log_norm(state.bandwidth, state.n_features_in, state.kernel)
    if state.sample_weight is None:
        total_weight = float(state.training_data.shape[0])
    else:
        weights = np.asarray(state.sample_weight, dtype=np.float64)
        log_terms = log_terms + np.where(weights > 0.0, np.log(weights), -np.inf)[np.newaxis, :]
        total_weight = float(np.sum(weights))
    return np.asarray(_logsumexp_rows(log_terms) - math.log(total_weight), dtype=np.float64)


@register_atom(witness_kernel_density_score)
@icontract.require(lambda X: _matrix_2d(X), "X must be 2D")
@icontract.require(lambda X: _finite_matrix(X), "X must contain finite values")
@icontract.require(lambda state: _kernel_density_state_valid(state), "state must be a fitted dense kernel-density estimator")
@icontract.require(lambda X, state: _kernel_density_feature_count_matches(X, state), "X feature count must match fitted state")
@icontract.ensure(lambda result: _log_score_valid(result), "total log density must be non-NaN")
def kernel_density_score(X: NDArray[np.float64], state: KernelDensityState) -> float:
    """Compute total Euclidean kernel log density over query samples."""
    return float(np.sum(kernel_density_score_samples(X, state)))


@register_atom(witness_kernel_density_sample)
@icontract.require(lambda state: _kernel_density_sampling_state_valid(state), "state must support gaussian or tophat sampling")
@icontract.require(lambda n_samples: _positive_sample_count(n_samples), "n_samples must be positive")
@icontract.ensure(lambda result, state, n_samples: _kernel_density_sample_valid(result, state, n_samples), "samples must be finite with fitted feature count")
def kernel_density_sample(
    state: KernelDensityState,
    n_samples: int = 1,
    random_state: int | None = None,
) -> NDArray[np.float64]:
    """Generate samples from fitted gaussian or tophat kernel density."""
    rng = check_random_state(random_state)
    u = rng.uniform(0.0, 1.0, size=n_samples)
    if state.sample_weight is None:
        indices = (u * state.training_data.shape[0]).astype(np.int64)
    else:
        cumulative = np.cumsum(np.asarray(state.sample_weight, dtype=np.float64))
        indices = np.searchsorted(cumulative, u * cumulative[-1])
    if state.kernel == "gaussian":
        return np.asarray(np.atleast_2d(rng.normal(state.training_data[indices], state.bandwidth)), dtype=np.float64)
    raw = rng.normal(size=(n_samples, state.n_features_in))
    squared_norm = np.sum(raw * raw, axis=1)
    correction = np.zeros(n_samples, dtype=np.float64)
    nonzero = squared_norm > 0.0
    correction[nonzero] = (
        gammainc(0.5 * state.n_features_in, 0.5 * squared_norm[nonzero]) ** (1.0 / state.n_features_in)
        * state.bandwidth
        / np.sqrt(squared_norm[nonzero])
    )
    return np.asarray(state.training_data[indices] + raw * correction[:, np.newaxis], dtype=np.float64)


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
