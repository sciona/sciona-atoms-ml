"""Ghost witnesses for agglomerative connectivity preprocessing atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def _check_matrix(values: AbstractArray, name: str) -> tuple[int, int]:
    if len(values.shape) != 2:
        raise ValueError(f"{name} must be 2D")
    rows, cols = int(values.shape[0]), int(values.shape[1])
    if rows < 1 or cols < 1:
        raise ValueError(f"{name} must be nonempty")
    return rows, cols


def witness_agglomerative_fix_connected_components(
    X: AbstractArray,
    graph: AbstractArray,
    n_connected_components: int,
    component_labels: AbstractArray,
    *,
    mode: str = "distance",
    metric: str = "euclidean",
) -> AbstractArray:
    """Describe graph completion across disconnected agglomerative components."""
    del metric
    n_samples, n_features = _check_matrix(X, "X")
    del n_features
    graph_rows, graph_cols = _check_matrix(graph, "graph")
    if graph_rows != graph_cols:
        raise ValueError("graph must be square")
    if graph_rows != n_samples:
        raise ValueError("graph sample count must match X")
    if n_connected_components < 1:
        raise ValueError("n_connected_components must be positive")
    if component_labels.shape != (n_samples,):
        raise ValueError("component_labels must match the sample count")
    if mode not in {"connectivity", "distance"}:
        raise ValueError("mode must be connectivity or distance")
    return AbstractArray(shape=(n_samples, n_samples), dtype="float64")


def witness_agglomerative_fix_connectivity(
    X: AbstractArray,
    connectivity: AbstractArray,
    affinity: str,
) -> tuple[AbstractArray, int]:
    """Describe normalized connectivity preprocessing before agglomerative tree building."""
    del affinity
    n_samples, _ = _check_matrix(X, "X")
    conn_rows, conn_cols = _check_matrix(connectivity, "connectivity")
    if conn_rows != conn_cols:
        raise ValueError("connectivity must be square")
    if conn_rows != n_samples:
        raise ValueError("connectivity must match the sample count in X")
    return AbstractArray(shape=(n_samples, n_samples), dtype="float64"), 1
