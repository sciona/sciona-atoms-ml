"""Agglomerative connectivity preprocessing atoms adapted from scikit-learn."""

from __future__ import annotations

import warnings

import icontract
import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray
from scipy.sparse.csgraph import connected_components

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_agglomerative_fix_connected_components,
    witness_agglomerative_fix_connectivity,
)

MatrixLike = NDArray[np.float64] | sp.spmatrix | list[list[float]]
ConnectivityLike = sp.spmatrix | NDArray[np.float64] | list[list[float]]
LabelsLike = NDArray[np.int64] | NDArray[np.int32] | list[int]

def _is_2d_matrix(values: MatrixLike) -> bool:
    if sp.issparse(values):
        return bool(values.ndim == 2 and values.shape[0] >= 1 and values.shape[1] >= 1)
    try:
        matrix = np.asarray(values)
    except (TypeError, ValueError):
        return False
    return bool(matrix.ndim == 2 and matrix.shape[0] >= 1 and matrix.shape[1] >= 1)

def _sample_count(values: MatrixLike) -> int:
    return int(values.shape[0]) if sp.issparse(values) else int(np.asarray(values).shape[0])

def _shape(values: MatrixLike) -> tuple[int, int]:
    return tuple(values.shape) if sp.issparse(values) else tuple(np.asarray(values).shape)

def _sparse_square_graph(graph: sp.spmatrix) -> bool:
    return bool(sp.issparse(graph) and graph.ndim == 2 and graph.shape[0] >= 1 and graph.shape[0] == graph.shape[1])

def _connectivity_matches_samples(X: MatrixLike, connectivity: ConnectivityLike) -> bool:
    if not _is_2d_matrix(X):
        return False
    try:
        shape = _shape(connectivity)
    except (TypeError, ValueError):
        return False
    n_samples = _sample_count(X)
    return bool(len(shape) == 2 and shape == (n_samples, n_samples))

def _positive_int(value: int) -> bool:
    return bool(isinstance(value, int) and not isinstance(value, bool) and value >= 1)

def _component_labels_valid(
    component_labels: LabelsLike,
    graph: sp.spmatrix,
    n_connected_components: int,
) -> bool:
    if not (_sparse_square_graph(graph) and _positive_int(n_connected_components)):
        return False
    labels = np.asarray(component_labels)
    if labels.ndim != 1 or labels.shape[0] != graph.shape[0]:
        return False
    if labels.dtype.kind not in {"i", "u"}:
        return False
    unique = np.unique(labels)
    return bool(unique.shape[0] == n_connected_components and np.array_equal(unique, np.arange(n_connected_components)))

def _metric_valid(metric: str) -> bool:
    return bool(isinstance(metric, str) and metric != "")

def _mode_valid(mode: str) -> bool:
    return bool(isinstance(mode, str) and mode in {"connectivity", "distance"})

def _precomputed_shape_valid(X: MatrixLike, metric: str) -> bool:
    if metric != "precomputed":
        return True
    return bool(_is_2d_matrix(X) and _shape(X)[0] == _shape(X)[1])

def _completed_graph_valid(result: sp.spmatrix, graph: sp.spmatrix) -> bool:
    return bool(sp.issparse(result) and result.shape == graph.shape)

def _fixed_connectivity_valid(result: tuple[sp.spmatrix, int], X: MatrixLike) -> bool:
    if not isinstance(result, tuple) or len(result) != 2:
        return False
    connectivity, n_connected_components = result
    n_samples = _sample_count(X)
    return bool(
        sp.isspmatrix_lil(connectivity)
        and connectivity.shape == (n_samples, n_samples)
        and isinstance(n_connected_components, (int, np.integer))
        and 1 <= int(n_connected_components) <= n_samples
    )

@register_atom(witness_agglomerative_fix_connected_components)
@icontract.require(lambda X: _is_2d_matrix(X), "X must be a nonempty 2D feature or distance matrix")
@icontract.require(lambda graph: _sparse_square_graph(graph), "graph must be a nonempty square sparse matrix")
@icontract.require(lambda X, graph: _sample_count(X) == graph.shape[0], "X and graph must describe the same samples")
@icontract.require(
    lambda component_labels, graph, n_connected_components: _component_labels_valid(component_labels, graph, n_connected_components),
    "component_labels must be a contiguous component partition for the graph",
)
@icontract.require(lambda metric: _metric_valid(metric), "metric must be a nonempty string")
@icontract.require(lambda mode: _mode_valid(mode), "mode must be 'connectivity' or 'distance'")
@icontract.require(lambda X, metric: _precomputed_shape_valid(X, metric), "precomputed metric requires a square distance matrix")
@icontract.ensure(lambda result, graph: _completed_graph_valid(result, graph), "completed graph must remain sparse with the same shape")
def agglomerative_fix_connected_components(
    X: MatrixLike,
    graph: sp.spmatrix,
    n_connected_components: int,
    component_labels: LabelsLike,
    *,
    mode: str = "distance",
    metric: str = "euclidean",
) -> sp.spmatrix:
    from sklearn.metrics import pairwise_distances
    """Connect disjoint graph components using sklearn's nearest cross-component rule."""
    if metric == "precomputed" and sp.issparse(X):
        raise RuntimeError(
            "_fix_connected_components with metric='precomputed' requires the "
            "full distance matrix in X, and does not work with a sparse "
            "neighbors graph."
        )

    completed_graph = graph.copy()
    labels = np.asarray(component_labels)

    for i in range(n_connected_components):
        idx_i = np.flatnonzero(labels == i)
        Xi = X[idx_i] if sp.issparse(X) else np.asarray(X)[idx_i]
        for j in range(i):
            idx_j = np.flatnonzero(labels == j)
            Xj = X[idx_j] if sp.issparse(X) else np.asarray(X)[idx_j]

            if metric == "precomputed":
                distances = np.asarray(X)[np.ix_(idx_i, idx_j)]
            else:
                distances = pairwise_distances(Xi, Xj, metric=metric)

            ii, jj = np.unravel_index(int(np.argmin(distances, axis=None)), distances.shape)
            if mode == "connectivity":
                completed_graph[idx_i[ii], idx_j[jj]] = 1
                completed_graph[idx_j[jj], idx_i[ii]] = 1
            elif mode == "distance":
                completed_graph[idx_i[ii], idx_j[jj]] = distances[ii, jj]
                completed_graph[idx_j[jj], idx_i[ii]] = distances[ii, jj]
            else:
                raise ValueError(
                    "Unknown mode=%r, should be one of ['connectivity', 'distance']."
                    % mode
                )

    return completed_graph

@register_atom(witness_agglomerative_fix_connectivity)
@icontract.require(lambda X: _is_2d_matrix(X), "X must be a nonempty 2D feature or distance matrix")
@icontract.require(lambda X, connectivity: _connectivity_matches_samples(X, connectivity), "connectivity must be square and match the sample count in X")
@icontract.require(lambda affinity: _metric_valid(affinity), "affinity must be a nonempty string")
@icontract.require(lambda X, affinity: _precomputed_shape_valid(X, affinity), "precomputed affinity requires a square distance matrix")
@icontract.ensure(lambda result, X: _fixed_connectivity_valid(result, X), "fixed connectivity must be a LIL sparse matrix with a valid component count")
def agglomerative_fix_connectivity(
    X: MatrixLike,
    connectivity: ConnectivityLike,
    affinity: str,
) -> tuple[sp.spmatrix, int]:
    """Normalize, count, and complete an agglomerative connectivity matrix."""
    fixed_connectivity = connectivity + connectivity.T
    if not sp.issparse(fixed_connectivity):
        fixed_connectivity = sp.lil_matrix(fixed_connectivity)
    if fixed_connectivity.format != "lil":
        fixed_connectivity = fixed_connectivity.tolil()

    n_connected_components, labels = connected_components(fixed_connectivity)

    if n_connected_components > 1:
        warnings.warn(
            "the number of connected components of the "
            "connectivity matrix is %d > 1. Completing it to avoid "
            "stopping the tree early." % n_connected_components,
            stacklevel=2,
        )
        fixed_connectivity = agglomerative_fix_connected_components(
            X=X,
            graph=fixed_connectivity,
            n_connected_components=int(n_connected_components),
            component_labels=np.asarray(labels, dtype=np.int64),
            metric=affinity,
            mode="connectivity",
        )

    return fixed_connectivity, int(n_connected_components)
