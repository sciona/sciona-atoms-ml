"""Ghost witnesses for DBSCAN output-packaging helpers."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_dbscan_core_sample_indices(core_samples: AbstractArray) -> AbstractArray:
    """Describe DBSCAN core-sample indices from a 1D mask."""
    if len(core_samples.shape) != 1 or int(core_samples.shape[0]) < 1:
        raise ValueError("core_samples must be a nonempty 1D vector")
    return AbstractArray(shape=(None,), dtype="int64", min_val=0)


def witness_dbscan_sparse_core_components(
    X: AbstractArray,
    core_sample_indices: AbstractArray,
) -> AbstractArray:
    """Describe sparse core-component rows selected from X."""
    if len(X.shape) != 2 or int(X.shape[0]) < 1 or int(X.shape[1]) < 1:
        raise ValueError("X must be a nonempty 2D matrix")
    if len(core_sample_indices.shape) != 1:
        raise ValueError("core_sample_indices must be 1D")
    return AbstractArray(shape=(int(core_sample_indices.shape[0]), int(X.shape[1])), dtype="float64")
