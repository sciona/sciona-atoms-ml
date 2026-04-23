"""BIRCH no-global-clustering atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import Birch as SklearnBirch
from sklearn.metrics import pairwise_distances_argmin
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils.validation import check_array

from sciona.ghost.registry import register_atom

from .state_models import BirchNoGlobalState
from .witnesses import (
    witness_birch_fit_no_global,
    witness_birch_predict_no_global,
    witness_birch_transform_no_global,
)

MatrixLike = NDArray[np.float64] | list[list[float]]


def _is_2d_finite_matrix(X: MatrixLike) -> bool:
    try:
        values = np.asarray(X, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(values.ndim == 2 and values.shape[0] >= 1 and values.shape[1] >= 1 and np.all(np.isfinite(values)))


def _positive_float(value: float) -> bool:
    return bool(isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(float(value)) and float(value) > 0.0)


def _branching_factor_valid(value: int) -> bool:
    return bool(isinstance(value, int) and not isinstance(value, bool) and value >= 2)


def _bool_value(value: bool) -> bool:
    return bool(isinstance(value, bool))


def _labels_valid(labels: NDArray[np.int_], X: MatrixLike) -> bool:
    values = np.asarray(labels)
    n_samples = np.asarray(X).shape[0]
    return bool(values.shape == (n_samples,) and np.issubdtype(values.dtype, np.integer) and np.all(values >= 0))


def _centers_valid(centers: NDArray[np.float64], n_features: int) -> bool:
    values = np.asarray(centers)
    return bool(
        values.ndim == 2
        and values.shape[0] >= 1
        and values.shape[1] == n_features
        and np.issubdtype(values.dtype, np.floating)
        and np.all(np.isfinite(values))
    )


def _state_valid(state: BirchNoGlobalState) -> bool:
    labels_ok = state.labels is None or (
        state.labels.ndim == 1
        and np.issubdtype(state.labels.dtype, np.integer)
        and np.all(state.labels >= 0)
    )
    return bool(
        isinstance(state, BirchNoGlobalState)
        and _centers_valid(state.subcluster_centers, state.n_features_in)
        and state.subcluster_labels.shape == (state.subcluster_centers.shape[0],)
        and np.issubdtype(state.subcluster_labels.dtype, np.integer)
        and np.array_equal(state.subcluster_labels, np.arange(state.subcluster_centers.shape[0]))
        and labels_ok
        and _positive_float(state.threshold)
        and _branching_factor_valid(state.branching_factor)
        and _bool_value(state.compute_labels)
        and state.n_features_out == state.subcluster_centers.shape[0]
    )


def _feature_count_matches(X: MatrixLike, state: BirchNoGlobalState) -> bool:
    return bool(_is_2d_finite_matrix(X) and _state_valid(state) and np.asarray(X).shape[1] == state.n_features_in)


def _transform_valid(result: NDArray[np.float64], X: MatrixLike, state: BirchNoGlobalState) -> bool:
    values = np.asarray(result)
    return bool(
        values.shape == (np.asarray(X).shape[0], state.n_features_out)
        and np.issubdtype(values.dtype, np.floating)
        and np.all(np.isfinite(values))
        and np.all(values >= 0.0)
    )


@register_atom(witness_birch_fit_no_global)
@icontract.require(lambda X: _is_2d_finite_matrix(X), "X must be a finite 2D matrix")
@icontract.require(lambda threshold: _positive_float(threshold), "threshold must be positive")
@icontract.require(lambda branching_factor: _branching_factor_valid(branching_factor), "branching_factor must be at least two")
@icontract.require(lambda compute_labels: _bool_value(compute_labels), "compute_labels must be boolean")
@icontract.ensure(lambda result: _state_valid(result), "BIRCH state must contain no-global subcluster centers and labels")
def birch_fit_no_global(
    X: MatrixLike,
    *,
    threshold: float = 0.5,
    branching_factor: int = 50,
    compute_labels: bool = True,
) -> BirchNoGlobalState:
    """Fit BIRCH with n_clusters=None and return immutable CF-tree summaries."""
    checked_x = np.asarray(check_array(X, dtype=np.float64, ensure_2d=True), dtype=np.float64)
    model = SklearnBirch(
        threshold=float(threshold),
        branching_factor=int(branching_factor),
        n_clusters=None,
        compute_labels=bool(compute_labels),
    ).fit(checked_x)
    labels = None if not compute_labels else np.asarray(model.labels_, dtype=np.int_).copy()
    return BirchNoGlobalState(
        subcluster_centers=np.asarray(model.subcluster_centers_, dtype=np.float64).copy(),
        subcluster_labels=np.asarray(model.subcluster_labels_, dtype=np.int_).copy(),
        labels=labels,
        threshold=float(threshold),
        branching_factor=int(branching_factor),
        compute_labels=bool(compute_labels),
        n_features_in=int(model.n_features_in_),
        n_features_out=int(model._n_features_out),
    )


@register_atom(witness_birch_predict_no_global)
@icontract.require(lambda X: _is_2d_finite_matrix(X), "X must be a finite 2D matrix")
@icontract.require(lambda X, state: _feature_count_matches(X, state), "X feature count must match BIRCH state")
@icontract.ensure(lambda result, X: _labels_valid(result, X), "predicted labels must match sample count")
def birch_predict_no_global(
    X: MatrixLike,
    state: BirchNoGlobalState,
) -> NDArray[np.int_]:
    """Predict labels by nearest BIRCH subcluster center without global clustering."""
    checked_x = np.asarray(check_array(X, dtype=np.float64, ensure_2d=True), dtype=np.float64)
    argmin = pairwise_distances_argmin(checked_x, state.subcluster_centers)
    return np.asarray(state.subcluster_labels[argmin], dtype=np.int_)


@register_atom(witness_birch_transform_no_global)
@icontract.require(lambda X: _is_2d_finite_matrix(X), "X must be a finite 2D matrix")
@icontract.require(lambda X, state: _feature_count_matches(X, state), "X feature count must match BIRCH state")
@icontract.ensure(lambda result, X, state: _transform_valid(result, X, state), "distance matrix must match sample and subcluster counts")
def birch_transform_no_global(
    X: MatrixLike,
    state: BirchNoGlobalState,
) -> NDArray[np.float64]:
    """Return distances from samples to BIRCH no-global subcluster centers."""
    checked_x = np.asarray(check_array(X, dtype=np.float64, ensure_2d=True), dtype=np.float64)
    return np.asarray(euclidean_distances(checked_x, state.subcluster_centers), dtype=np.float64)
