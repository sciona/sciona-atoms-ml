"""Dense neighbors graph atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray
from sklearn.utils.validation import check_array

from sciona.ghost.registry import register_atom

from .state_models import NeighborsGraphTransformerState
from .witnesses import (
    witness_kneighbors_graph,
    witness_kneighbors_transform,
    witness_kneighbors_transformer_fit,
    witness_radius_neighbors_graph,
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
