"""Ghost witnesses for spectral biclustering piecewise and projection atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def _check_vector(values: AbstractArray, name: str) -> int:
    if len(values.shape) != 1:
        raise ValueError(f"{name} must be one-dimensional")
    length = int(values.shape[0])
    if length < 1:
        raise ValueError(f"{name} must be nonempty")
    return length


def _check_matrix(values: AbstractArray, name: str) -> tuple[int, int]:
    if len(values.shape) != 2:
        raise ValueError(f"{name} must be two-dimensional")
    rows, cols = int(values.shape[0]), int(values.shape[1])
    if rows < 1 or cols < 1:
        raise ValueError(f"{name} must be nonempty")
    return rows, cols


def witness_bicluster_piecewise_vector(
    centroids: AbstractArray,
    labels: AbstractArray,
) -> AbstractArray:
    """Describe rebuilding one piecewise vector from centers and labels."""
    _, cols = _check_matrix(centroids, "centroids")
    length = _check_vector(labels, "labels")
    return AbstractArray(shape=(length * cols,), dtype="float64")


def witness_bicluster_piecewise_residual_norms(
    vectors: AbstractArray,
    piecewise_vectors: AbstractArray,
) -> AbstractArray:
    """Describe rowwise residual sizes between vectors and piecewise fits."""
    rows, cols = _check_matrix(vectors, "vectors")
    piece_rows, piece_cols = _check_matrix(piecewise_vectors, "piecewise_vectors")
    if rows != piece_rows or cols != piece_cols:
        raise ValueError("vectors and piecewise_vectors must share the same shape")
    return AbstractArray(shape=(rows,), dtype="float64", min_val=0)


def witness_bicluster_select_best_piecewise_vectors(
    vectors: AbstractArray,
    residual_norms: AbstractArray,
    n_best: int,
) -> AbstractArray:
    """Describe selecting the rows with the smallest residual sizes."""
    rows, cols = _check_matrix(vectors, "vectors")
    residual_rows = _check_vector(residual_norms, "residual_norms")
    if residual_rows != rows:
        raise ValueError("residual_norms must match the vector count")
    if n_best < 1 or n_best > rows:
        raise ValueError("n_best must be between one and the number of vectors")
    return AbstractArray(shape=(n_best, cols), dtype="float64")


def witness_bicluster_project_dense(
    data: AbstractArray,
    vectors: AbstractArray,
) -> AbstractArray:
    """Describe dense projection before clustering."""
    rows, data_cols = _check_matrix(data, "data")
    vector_rows, vector_cols = _check_matrix(vectors, "vectors")
    if data_cols != vector_rows:
        raise ValueError("data and vectors must have aligned inner dimensions")
    return AbstractArray(shape=(rows, vector_cols), dtype="float64")
