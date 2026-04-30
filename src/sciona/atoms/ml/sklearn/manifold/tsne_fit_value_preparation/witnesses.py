"""Ghost witnesses for sklearn t-SNE fit-value preparation helpers."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def _square_shape(values: AbstractArray, name: str) -> int:
    if len(values.shape) != 2:
        raise ValueError(f"{name} must be 2D")
    rows, cols = int(values.shape[0]), int(values.shape[1])
    if rows != cols or rows < 2:
        raise ValueError(f"{name} must be square with at least two rows")
    return rows


def _matrix_shape(values: AbstractArray, name: str) -> tuple[int, int]:
    if len(values.shape) != 2:
        raise ValueError(f"{name} must be 2D")
    rows, cols = int(values.shape[0]), int(values.shape[1])
    if rows < 1 or cols < 1:
        raise ValueError(f"{name} must be nonempty")
    return rows, cols


def _vector_len(values: AbstractArray, name: str) -> int:
    if len(values.shape) != 1:
        raise ValueError(f"{name} must be a vector")
    length = int(values.shape[0])
    if length < 1:
        raise ValueError(f"{name} must be nonempty")
    return length


def witness_tsne_exact_distance_matrix(
    distances: AbstractArray,
    *,
    metric: str,
) -> AbstractArray:
    """Describe exact-method distance postprocessing in t-SNE."""
    del metric
    n = _square_shape(distances, "distances")
    return AbstractArray(shape=(n, n), dtype="float64")


def witness_tsne_neighbor_graph_squared_data(
    data: AbstractArray,
) -> AbstractArray:
    """Describe squared Barnes-Hut neighbor-graph distance values."""
    return AbstractArray(shape=(_vector_len(data, "data"),), dtype="float64")


def witness_tsne_exact_probability_vector(
    P: AbstractArray,
) -> AbstractArray:
    """Describe exact-method condensed probability validation in t-SNE."""
    return AbstractArray(shape=(_vector_len(P, "P"),), dtype="float64")


def witness_tsne_provided_layout_matrix(
    init: AbstractArray,
) -> AbstractArray:
    """Describe passthrough of a provided starting matrix."""
    return AbstractArray(shape=_matrix_shape(init, "init"), dtype="float64")
