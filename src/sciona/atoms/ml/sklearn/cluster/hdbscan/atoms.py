"""Limited HDBSCAN public-boundary atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import HDBSCAN as SklearnHDBSCAN

from sciona.ghost.registry import register_atom

from .state_models import HDBSCANState
from .witnesses import witness_hdbscan_fit, witness_hdbscan_fit_predict

MatrixLike = NDArray[np.float64] | list[list[float]]

_VALID_METRICS = {"euclidean", "l2", "minkowski", "manhattan", "cityblock"}
_VALID_ALGORITHMS = {"auto", "brute", "kd_tree", "ball_tree"}
_VALID_SELECTION_METHODS = {"eom", "leaf"}
_TREE_FIELDS = ("left_node", "right_node", "value", "cluster_size")


def _is_dense_finite_matrix(X: MatrixLike) -> bool:
    try:
        values = np.asarray(X, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(values.ndim == 2 and values.shape[0] >= 2 and values.shape[1] >= 1 and np.all(np.isfinite(values)))


def _positive_int(value: int) -> bool:
    return bool(isinstance(value, int) and not isinstance(value, bool) and value >= 1)


def _min_cluster_size_valid(value: int, X: MatrixLike) -> bool:
    return bool(isinstance(value, int) and not isinstance(value, bool) and 2 <= value <= np.asarray(X).shape[0])


def _min_samples_valid(value: int | None, X: MatrixLike) -> bool:
    return bool(value is None or (_positive_int(value) and value <= np.asarray(X).shape[0]))


def _nonnegative_float(value: float) -> bool:
    return bool(isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(float(value)) and float(value) >= 0.0)


def _positive_float(value: float) -> bool:
    return bool(isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(float(value)) and float(value) > 0.0)


def _max_cluster_size_valid(value: int | None) -> bool:
    return bool(value is None or (isinstance(value, int) and not isinstance(value, bool) and value >= 0))


def _metric_valid(metric: str) -> bool:
    return bool(isinstance(metric, str) and metric in _VALID_METRICS)


def _metric_params_valid(metric_params: dict[str, float] | None) -> bool:
    if metric_params is None:
        return True
    return bool(
        isinstance(metric_params, dict)
        and all(isinstance(key, str) and isinstance(value, (int, float)) and np.isfinite(float(value)) for key, value in metric_params.items())
    )


def _algorithm_valid(algorithm: str) -> bool:
    return bool(isinstance(algorithm, str) and algorithm in _VALID_ALGORITHMS)


def _cluster_selection_method_valid(method: str) -> bool:
    return bool(isinstance(method, str) and method in _VALID_SELECTION_METHODS)


def _bool_value(value: bool) -> bool:
    return bool(isinstance(value, bool))


def _n_jobs_valid(n_jobs: int | None) -> bool:
    return bool(n_jobs is None or (isinstance(n_jobs, int) and not isinstance(n_jobs, bool) and n_jobs != 0))


def _labels_valid(labels: NDArray[np.int_], X: MatrixLike) -> bool:
    values = np.asarray(labels)
    n_samples = np.asarray(X).shape[0]
    return bool(
        values.shape == (n_samples,)
        and np.issubdtype(values.dtype, np.integer)
        and np.all(values >= -1)
        and np.all(values < n_samples)
    )


def _probabilities_valid(probabilities: NDArray[np.float64], labels: NDArray[np.int_]) -> bool:
    values = np.asarray(probabilities)
    return bool(
        values.shape == np.asarray(labels).shape
        and np.issubdtype(values.dtype, np.floating)
        and np.all(np.isfinite(values))
        and np.all(values >= 0.0)
        and np.all(values <= 1.0)
    )


def _single_linkage_tree_valid(tree: NDArray[np.generic], n_samples: int) -> bool:
    values = np.asarray(tree)
    if values.ndim != 1 or values.shape[0] != n_samples - 1 or values.dtype.names != _TREE_FIELDS:
        return False
    return bool(
        np.all(values["left_node"] >= 0)
        and np.all(values["right_node"] >= 0)
        and np.all(np.isfinite(values["value"]))
        and np.all(values["value"] >= 0.0)
        and np.all(values["cluster_size"] >= 2)
    )


def _state_valid(state: HDBSCANState) -> bool:
    return bool(
        isinstance(state, HDBSCANState)
        and state.labels.ndim == 1
        and np.issubdtype(state.labels.dtype, np.integer)
        and np.all(state.labels >= -1)
        and _probabilities_valid(state.probabilities, state.labels)
        and _single_linkage_tree_valid(state.single_linkage_tree, state.labels.shape[0])
        and state.min_cluster_size >= 2
        and _positive_int(state.min_samples)
        and _nonnegative_float(state.cluster_selection_epsilon)
        and _max_cluster_size_valid(state.max_cluster_size)
        and _metric_valid(state.metric)
        and _positive_float(state.alpha)
        and _algorithm_valid(state.algorithm)
        and _positive_int(state.leaf_size)
        and _cluster_selection_method_valid(state.cluster_selection_method)
        and _bool_value(state.allow_single_cluster)
        and _positive_int(state.n_features_in)
    )


def _fit_model(
    X: MatrixLike,
    *,
    min_cluster_size: int,
    min_samples: int | None,
    cluster_selection_epsilon: float,
    max_cluster_size: int | None,
    metric: str,
    metric_params: dict[str, float] | None,
    alpha: float,
    algorithm: str,
    leaf_size: int,
    n_jobs: int | None,
    cluster_selection_method: str,
    allow_single_cluster: bool,
    copy: bool,
) -> SklearnHDBSCAN:
    checked_x = np.asarray(X, dtype=np.float64)
    model = SklearnHDBSCAN(
        min_cluster_size=int(min_cluster_size),
        min_samples=None if min_samples is None else int(min_samples),
        cluster_selection_epsilon=float(cluster_selection_epsilon),
        max_cluster_size=max_cluster_size,
        metric=metric,
        metric_params=metric_params,
        alpha=float(alpha),
        algorithm=algorithm,
        leaf_size=int(leaf_size),
        n_jobs=n_jobs,
        cluster_selection_method=cluster_selection_method,
        allow_single_cluster=bool(allow_single_cluster),
        store_centers=None,
        copy=bool(copy),
    )
    return model.fit(checked_x)


@register_atom(witness_hdbscan_fit)
@icontract.require(lambda X: _is_dense_finite_matrix(X), "X must be a dense finite matrix with at least two samples")
@icontract.require(lambda min_cluster_size, X: _min_cluster_size_valid(min_cluster_size, X), "min_cluster_size must be between two and sample count")
@icontract.require(lambda min_samples, X: _min_samples_valid(min_samples, X), "min_samples must be None or fit the sample count")
@icontract.require(lambda cluster_selection_epsilon: _nonnegative_float(cluster_selection_epsilon), "cluster_selection_epsilon must be nonnegative")
@icontract.require(lambda max_cluster_size: _max_cluster_size_valid(max_cluster_size), "max_cluster_size must be nonnegative or None")
@icontract.require(lambda metric: _metric_valid(metric), "metric must be a supported HDBSCAN string metric")
@icontract.require(lambda metric_params: _metric_params_valid(metric_params), "metric_params must be a finite numeric dictionary or None")
@icontract.require(lambda alpha: _positive_float(alpha), "alpha must be positive")
@icontract.require(lambda algorithm: _algorithm_valid(algorithm), "algorithm must be a valid HDBSCAN backend")
@icontract.require(lambda leaf_size: _positive_int(leaf_size), "leaf_size must be positive")
@icontract.require(lambda n_jobs: _n_jobs_valid(n_jobs), "n_jobs must be nonzero or None")
@icontract.require(lambda cluster_selection_method: _cluster_selection_method_valid(cluster_selection_method), "cluster_selection_method must be 'eom' or 'leaf'")
@icontract.require(lambda allow_single_cluster: _bool_value(allow_single_cluster), "allow_single_cluster must be boolean")
@icontract.require(lambda copy: _bool_value(copy), "copy must be boolean")
@icontract.ensure(lambda result: _state_valid(result), "HDBSCAN state must contain labels, probabilities, and linkage tree")
def hdbscan_fit(
    X: MatrixLike,
    *,
    min_cluster_size: int = 5,
    min_samples: int | None = None,
    cluster_selection_epsilon: float = 0.0,
    max_cluster_size: int | None = None,
    metric: str = "euclidean",
    metric_params: dict[str, float] | None = None,
    alpha: float = 1.0,
    algorithm: str = "auto",
    leaf_size: int = 40,
    n_jobs: int | None = None,
    cluster_selection_method: str = "eom",
    allow_single_cluster: bool = False,
    copy: bool = True,
) -> HDBSCANState:
    """Fit HDBSCAN through sklearn's public boundary and return immutable state."""
    model = _fit_model(
        X,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,
        max_cluster_size=max_cluster_size,
        metric=metric,
        metric_params=metric_params,
        alpha=alpha,
        algorithm=algorithm,
        leaf_size=leaf_size,
        n_jobs=n_jobs,
        cluster_selection_method=cluster_selection_method,
        allow_single_cluster=allow_single_cluster,
        copy=copy,
    )
    return HDBSCANState(
        labels=np.asarray(model.labels_, dtype=np.int_).copy(),
        probabilities=np.asarray(model.probabilities_, dtype=np.float64).copy(),
        single_linkage_tree=np.asarray(model._single_linkage_tree_).copy(),
        min_cluster_size=int(min_cluster_size),
        min_samples=int(model._min_samples),
        cluster_selection_epsilon=float(cluster_selection_epsilon),
        max_cluster_size=max_cluster_size,
        metric=metric,
        alpha=float(alpha),
        algorithm=algorithm,
        leaf_size=int(leaf_size),
        cluster_selection_method=cluster_selection_method,
        allow_single_cluster=bool(allow_single_cluster),
        n_features_in=int(model.n_features_in_),
    )


@register_atom(witness_hdbscan_fit_predict)
@icontract.require(lambda X: _is_dense_finite_matrix(X), "X must be a dense finite matrix with at least two samples")
@icontract.require(lambda min_cluster_size, X: _min_cluster_size_valid(min_cluster_size, X), "min_cluster_size must be between two and sample count")
@icontract.require(lambda min_samples, X: _min_samples_valid(min_samples, X), "min_samples must be None or fit the sample count")
@icontract.require(lambda cluster_selection_epsilon: _nonnegative_float(cluster_selection_epsilon), "cluster_selection_epsilon must be nonnegative")
@icontract.require(lambda max_cluster_size: _max_cluster_size_valid(max_cluster_size), "max_cluster_size must be nonnegative or None")
@icontract.require(lambda metric: _metric_valid(metric), "metric must be a supported HDBSCAN string metric")
@icontract.require(lambda metric_params: _metric_params_valid(metric_params), "metric_params must be a finite numeric dictionary or None")
@icontract.require(lambda alpha: _positive_float(alpha), "alpha must be positive")
@icontract.require(lambda algorithm: _algorithm_valid(algorithm), "algorithm must be a valid HDBSCAN backend")
@icontract.require(lambda leaf_size: _positive_int(leaf_size), "leaf_size must be positive")
@icontract.require(lambda n_jobs: _n_jobs_valid(n_jobs), "n_jobs must be nonzero or None")
@icontract.require(lambda cluster_selection_method: _cluster_selection_method_valid(cluster_selection_method), "cluster_selection_method must be 'eom' or 'leaf'")
@icontract.require(lambda allow_single_cluster: _bool_value(allow_single_cluster), "allow_single_cluster must be boolean")
@icontract.require(lambda copy: _bool_value(copy), "copy must be boolean")
@icontract.ensure(lambda result, X: _labels_valid(result, X), "HDBSCAN fit_predict labels must match sample count")
def hdbscan_fit_predict(
    X: MatrixLike,
    *,
    min_cluster_size: int = 5,
    min_samples: int | None = None,
    cluster_selection_epsilon: float = 0.0,
    max_cluster_size: int | None = None,
    metric: str = "euclidean",
    metric_params: dict[str, float] | None = None,
    alpha: float = 1.0,
    algorithm: str = "auto",
    leaf_size: int = 40,
    n_jobs: int | None = None,
    cluster_selection_method: str = "eom",
    allow_single_cluster: bool = False,
    copy: bool = True,
) -> NDArray[np.int_]:
    """Return labels from sklearn HDBSCAN fit_predict for dense finite inputs."""
    state = hdbscan_fit(
        X,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,
        max_cluster_size=max_cluster_size,
        metric=metric,
        metric_params=metric_params,
        alpha=alpha,
        algorithm=algorithm,
        leaf_size=leaf_size,
        n_jobs=n_jobs,
        cluster_selection_method=cluster_selection_method,
        allow_single_cluster=allow_single_cluster,
        copy=copy,
    )
    return state.labels.copy()
