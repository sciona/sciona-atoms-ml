"""Limited DBSCAN public-boundary atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .state_models import DBSCANState
from .witnesses import witness_dbscan_core_labels, witness_dbscan_fit

MatrixLike = NDArray[np.float64] | list[list[float]]
VectorLike = NDArray[np.float64] | list[float]

_VALID_METRICS = {"euclidean", "minkowski", "manhattan", "precomputed"}
_VALID_ALGORITHMS = {"auto", "ball_tree", "kd_tree", "brute"}

def _is_2d_matrix(X: MatrixLike) -> bool:
    try:
        values = np.asarray(X, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(values.ndim == 2 and values.shape[0] >= 1 and values.shape[1] >= 1 and np.all(np.isfinite(values)))

def _dbscan_input_valid(X: MatrixLike, metric: str) -> bool:
    if not _is_2d_matrix(X):
        return False
    values = np.asarray(X, dtype=np.float64)
    if metric == "precomputed":
        return bool(values.shape[0] == values.shape[1] and np.all(values >= 0.0))
    return True

def _positive_float(value: float) -> bool:
    return bool(isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(float(value)) and float(value) > 0.0)

def _positive_int(value: int) -> bool:
    return bool(isinstance(value, int) and not isinstance(value, bool) and value >= 1)

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

def _positive_float_or_none(value: float | None) -> bool:
    return bool(value is None or _positive_float(value))

def _n_jobs_valid(n_jobs: int | None) -> bool:
    return bool(n_jobs is None or (isinstance(n_jobs, int) and not isinstance(n_jobs, bool) and n_jobs != 0))

def _sample_weight_valid(sample_weight: VectorLike | None, X: MatrixLike) -> bool:
    if sample_weight is None:
        return True
    try:
        weights = np.asarray(sample_weight, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(weights.ndim == 1 and weights.shape[0] == np.asarray(X).shape[0] and np.all(np.isfinite(weights)))

def _labels_valid(labels: NDArray[np.int_], X: MatrixLike) -> bool:
    values = np.asarray(labels)
    n_samples = np.asarray(X).shape[0]
    return bool(
        values.shape == (n_samples,)
        and np.issubdtype(values.dtype, np.integer)
        and np.all(values >= -1)
        and np.all(values < n_samples)
    )

def _core_indices_valid(indices: NDArray[np.int_], X: MatrixLike) -> bool:
    values = np.asarray(indices)
    n_samples = np.asarray(X).shape[0]
    return bool(
        values.ndim == 1
        and np.issubdtype(values.dtype, np.integer)
        and np.all(values >= 0)
        and np.all(values < n_samples)
        and np.unique(values).shape[0] == values.shape[0]
    )

def _state_valid(state: DBSCANState) -> bool:
    return bool(
        isinstance(state, DBSCANState)
        and state.labels.ndim == 1
        and np.issubdtype(state.labels.dtype, np.integer)
        and np.all(state.labels >= -1)
        and _core_indices_valid(state.core_sample_indices, state.labels.reshape(-1, 1))
        and state.components.ndim == 2
        and state.components.shape[0] == state.core_sample_indices.shape[0]
        and _positive_float(state.eps)
        and _positive_int(state.min_samples)
        and _metric_valid(state.metric)
        and _algorithm_valid(state.algorithm)
        and _positive_int(state.leaf_size)
        and _positive_float_or_none(state.p)
        and _positive_int(state.n_features_in)
    )

def _fit_model(
    X: MatrixLike,
    *,
    eps: float,
    min_samples: int,
    metric: str,
    metric_params: dict[str, float] | None,
    algorithm: str,
    leaf_size: int,
    p: float | None,
    sample_weight: VectorLike | None,
    n_jobs: int | None,
) -> SklearnDBSCAN:
    from sklearn.cluster import DBSCAN as SklearnDBSCAN
    from sklearn.utils.validation import check_array
    checked_x = np.asarray(check_array(X, dtype=np.float64), dtype=np.float64)
    weights = None if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
    model = SklearnDBSCAN(
        eps=float(eps),
        min_samples=int(min_samples),
        metric=metric,
        metric_params=metric_params,
        algorithm=algorithm,
        leaf_size=int(leaf_size),
        p=p,
        n_jobs=n_jobs,
    )
    return model.fit(checked_x, sample_weight=weights)

@register_atom(witness_dbscan_fit)
@icontract.require(lambda metric: _metric_valid(metric), "metric must be one of the supported DBSCAN metrics")
@icontract.require(lambda X, metric: _dbscan_input_valid(X, metric), "X must be finite and compatible with the metric")
@icontract.require(lambda eps: _positive_float(eps), "eps must be positive")
@icontract.require(lambda min_samples: _positive_int(min_samples), "min_samples must be positive")
@icontract.require(lambda metric_params: _metric_params_valid(metric_params), "metric_params must be a finite numeric dictionary or None")
@icontract.require(lambda algorithm: _algorithm_valid(algorithm), "algorithm must be a valid nearest-neighbor backend")
@icontract.require(lambda leaf_size: _positive_int(leaf_size), "leaf_size must be positive")
@icontract.require(lambda p: _positive_float_or_none(p), "p must be positive or None")
@icontract.require(lambda sample_weight, X: _sample_weight_valid(sample_weight, X), "sample_weight must match sample count")
@icontract.require(lambda n_jobs: _n_jobs_valid(n_jobs), "n_jobs must be nonzero or None")
@icontract.ensure(lambda result: _state_valid(result), "DBSCAN state must contain labels and core sample metadata")
def dbscan_fit(
    X: MatrixLike,
    *,
    eps: float = 0.5,
    min_samples: int = 5,
    metric: str = "minkowski",
    metric_params: dict[str, float] | None = None,
    algorithm: str = "auto",
    leaf_size: int = 30,
    p: float | None = 2.0,
    sample_weight: VectorLike | None = None,
    n_jobs: int | None = None,
) -> DBSCANState:
    """Fit DBSCAN through sklearn's public boundary and return immutable state."""
    model = _fit_model(
        X,
        eps=eps,
        min_samples=min_samples,
        metric=metric,
        metric_params=metric_params,
        algorithm=algorithm,
        leaf_size=leaf_size,
        p=p,
        sample_weight=sample_weight,
        n_jobs=n_jobs,
    )
    return DBSCANState(
        core_sample_indices=np.asarray(model.core_sample_indices_, dtype=np.int_).copy(),
        labels=np.asarray(model.labels_, dtype=np.int_).copy(),
        components=np.asarray(model.components_, dtype=np.float64).copy(),
        eps=float(eps),
        min_samples=int(min_samples),
        metric=metric,
        algorithm=algorithm,
        leaf_size=int(leaf_size),
        p=None if p is None else float(p),
        n_features_in=int(model.n_features_in_),
    )

@register_atom(witness_dbscan_core_labels)
@icontract.require(lambda metric: _metric_valid(metric), "metric must be one of the supported DBSCAN metrics")
@icontract.require(lambda X, metric: _dbscan_input_valid(X, metric), "X must be finite and compatible with the metric")
@icontract.require(lambda eps: _positive_float(eps), "eps must be positive")
@icontract.require(lambda min_samples: _positive_int(min_samples), "min_samples must be positive")
@icontract.require(lambda metric_params: _metric_params_valid(metric_params), "metric_params must be a finite numeric dictionary or None")
@icontract.require(lambda algorithm: _algorithm_valid(algorithm), "algorithm must be a valid nearest-neighbor backend")
@icontract.require(lambda leaf_size: _positive_int(leaf_size), "leaf_size must be positive")
@icontract.require(lambda p: _positive_float_or_none(p), "p must be positive or None")
@icontract.require(lambda sample_weight, X: _sample_weight_valid(sample_weight, X), "sample_weight must match sample count")
@icontract.require(lambda n_jobs: _n_jobs_valid(n_jobs), "n_jobs must be nonzero or None")
@icontract.ensure(lambda result, X: _core_indices_valid(result[0], X) and _labels_valid(result[1], X), "DBSCAN helper output must contain core indices and labels")
def dbscan_core_labels(
    X: MatrixLike,
    *,
    eps: float = 0.5,
    min_samples: int = 5,
    metric: str = "minkowski",
    metric_params: dict[str, float] | None = None,
    algorithm: str = "auto",
    leaf_size: int = 30,
    p: float | None = 2.0,
    sample_weight: VectorLike | None = None,
    n_jobs: int | None = None,
) -> tuple[NDArray[np.int_], NDArray[np.int_]]:
    from sklearn.cluster import dbscan as sklearn_dbscan
    from sklearn.utils.validation import check_array
    """Return sklearn DBSCAN public-helper core sample indices and labels."""
    checked_x = np.asarray(check_array(X, dtype=np.float64), dtype=np.float64)
    weights = None if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
    core_indices, labels = sklearn_dbscan(
        checked_x,
        eps=float(eps),
        min_samples=int(min_samples),
        metric=metric,
        metric_params=metric_params,
        algorithm=algorithm,
        leaf_size=int(leaf_size),
        p=p,
        sample_weight=weights,
        n_jobs=n_jobs,
    )
    return np.asarray(core_indices, dtype=np.int_).copy(), np.asarray(labels, dtype=np.int_).copy()
