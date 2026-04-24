"""Ghost witnesses for spectral biclustering structure atoms."""

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


def witness_cocluster_singular_vector_count(n_clusters: int) -> int:
    """Describe the singular-vector count used by spectral coclustering."""
    if n_clusters < 1:
        raise ValueError("n_clusters must be positive")
    return 1


def witness_cocluster_stacked_embedding(
    row_diag: AbstractArray,
    u: AbstractArray,
    col_diag: AbstractArray,
    v: AbstractArray,
) -> AbstractArray:
    """Describe the stacked coclustering embedding sent to KMeans."""
    n_rows = _check_vector(row_diag, "row_diag")
    u_rows, n_components = _check_matrix(u, "u")
    n_cols = _check_vector(col_diag, "col_diag")
    v_rows, v_components = _check_matrix(v, "v")
    if u_rows != n_rows:
        raise ValueError("row_diag must match the row count of u")
    if v_rows != n_cols:
        raise ValueError("col_diag must match the row count of v")
    if v_components != n_components:
        raise ValueError("u and v must share the component count")
    return AbstractArray(shape=(n_rows + n_cols, n_components), dtype="float64")


def witness_cocluster_split_labels(labels: AbstractArray, n_rows: int) -> tuple[AbstractArray, AbstractArray]:
    """Describe splitting stacked coclustering labels into row and column labels."""
    total = _check_vector(labels, "labels")
    if n_rows < 1 or n_rows >= total:
        raise ValueError("n_rows must lie inside the label vector")
    return (
        AbstractArray(shape=(n_rows,), dtype="int64", min_val=0),
        AbstractArray(shape=(total - n_rows,), dtype="int64", min_val=0),
    )


def witness_cocluster_indicator_matrix(labels: AbstractArray, n_clusters: int) -> AbstractArray:
    """Describe the boolean indicator matrix for one label vector."""
    n_samples = _check_vector(labels, "labels")
    if n_clusters < 1:
        raise ValueError("n_clusters must be positive")
    return AbstractArray(shape=(n_clusters, n_samples), dtype="bool")


def witness_bicluster_effective_svd_dims(method: str, n_components: int) -> tuple[int, int]:
    """Describe the requested and discarded singular-vector counts."""
    del method
    if n_components < 1:
        raise ValueError("n_components must be positive")
    return 1, 0


def witness_bicluster_resolve_cluster_counts(
    n_clusters: int | tuple[int, int],
) -> tuple[int, int]:
    """Describe the row and column cluster counts for spectral biclustering."""
    del n_clusters
    return 1, 1


def witness_bicluster_indicator_grid(
    row_labels: AbstractArray,
    column_labels: AbstractArray,
    n_row_clusters: int,
    n_col_clusters: int,
) -> tuple[AbstractArray, AbstractArray]:
    """Describe the repeated row/column indicator grid for checkerboard biclusters."""
    n_rows = _check_vector(row_labels, "row_labels")
    n_cols = _check_vector(column_labels, "column_labels")
    if n_row_clusters < 1 or n_col_clusters < 1:
        raise ValueError("cluster counts must be positive")
    total = n_row_clusters * n_col_clusters
    return (
        AbstractArray(shape=(total, n_rows), dtype="bool"),
        AbstractArray(shape=(total, n_cols), dtype="bool"),
    )
