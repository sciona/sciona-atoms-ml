"""Ghost witnesses for sklearn DBSCAN fit-bookkeeping helpers."""

from __future__ import annotations

from scipy.sparse import csr_matrix

from sciona.ghost.abstract import AbstractArray


def witness_dbscan_precomputed_sparse_self_neighbors(X: AbstractArray) -> csr_matrix:
    """Describe a sparse precomputed distance graph with explicit self-neighbor entries."""
    if len(X.shape) != 2:
        raise ValueError("X must be 2D")
    n_rows, n_cols = int(X.shape[0]), int(X.shape[1])
    if n_rows < 1 or n_cols < 1 or n_rows != n_cols:
        raise ValueError("X must be a nonempty square matrix")
    return csr_matrix((n_rows, n_cols), dtype=float)


def witness_dbscan_neighbor_count_vector(neighborhoods: tuple[AbstractArray, ...]) -> AbstractArray:
    """Describe per-sample DBSCAN neighborhood counts."""
    n_samples = len(neighborhoods)
    if n_samples < 1:
        raise ValueError("neighborhoods must be nonempty")
    for block in neighborhoods:
        if len(block.shape) != 1:
            raise ValueError("each neighborhood must be a 1D index vector")
    return AbstractArray(shape=(n_samples,), dtype="int64", min_val=0)


def witness_dbscan_weighted_neighbor_sums(
    neighborhoods: tuple[AbstractArray, ...],
    sample_weight: AbstractArray,
) -> AbstractArray:
    """Describe per-sample DBSCAN weighted neighborhood mass."""
    if len(sample_weight.shape) != 1:
        raise ValueError("sample_weight must be 1D")
    n_samples = len(neighborhoods)
    if n_samples < 1 or int(sample_weight.shape[0]) != n_samples:
        raise ValueError("sample_weight must match the number of neighborhoods")
    for block in neighborhoods:
        if len(block.shape) != 1:
            raise ValueError("each neighborhood must be a 1D index vector")
    return AbstractArray(shape=(n_samples,), dtype="float64")


def witness_dbscan_core_sample_mask(neighbor_mass: AbstractArray, min_samples: int) -> AbstractArray:
    """Describe the uint8 core-sample mask from neighborhood mass and min_samples."""
    if len(neighbor_mass.shape) != 1:
        raise ValueError("neighbor_mass must be 1D")
    if int(neighbor_mass.shape[0]) < 1:
        raise ValueError("neighbor_mass must be nonempty")
    if min_samples < 1:
        raise ValueError("min_samples must be positive")
    return AbstractArray(shape=(int(neighbor_mass.shape[0]),), dtype="uint8", min_val=0, max_val=1)


def witness_dbscan_initial_noise_labels(n_samples: int) -> AbstractArray:
    """Describe DBSCAN's initial all-noise label vector."""
    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    return AbstractArray(shape=(n_samples,), dtype="int64", min_val=-1, max_val=-1)


def witness_dbscan_dense_core_components(X: AbstractArray, core_sample_indices: AbstractArray) -> AbstractArray:
    """Describe dense core-sample component extraction."""
    if len(X.shape) != 2:
        raise ValueError("X must be 2D")
    n_samples, n_features = int(X.shape[0]), int(X.shape[1])
    if n_samples < 1 or n_features < 1:
        raise ValueError("X must be nonempty")
    if len(core_sample_indices.shape) != 1:
        raise ValueError("core_sample_indices must be 1D")
    n_core = int(core_sample_indices.shape[0])
    return AbstractArray(shape=(n_core, n_features), dtype="float64")


def witness_dbscan_empty_components(n_features: int) -> AbstractArray:
    """Describe DBSCAN's empty-components fallback."""
    if n_features < 1:
        raise ValueError("n_features must be positive")
    return AbstractArray(shape=(0, n_features), dtype="float64")
