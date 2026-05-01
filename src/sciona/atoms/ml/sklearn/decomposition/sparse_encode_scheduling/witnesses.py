"""Ghost witnesses for sparse-encode scheduling atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_sparse_encode_parallel_required(
    effective_job_count: int,
    algorithm: str,
) -> AbstractArray:
    """Describe the Boolean sparse_encode parallel-branch predicate."""
    del effective_job_count, algorithm
    return AbstractArray(shape=(), dtype="bool")


def witness_sparse_encode_sample_bounds(
    n_samples: int,
    effective_job_count: int,
) -> AbstractArray:
    """Describe sklearn's even sparse_encode sample-slice bounds."""
    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    if effective_job_count < 1:
        raise ValueError("effective_job_count must be positive")
    return AbstractArray(shape=(effective_job_count, 2), dtype="int64")


def witness_sparse_encode_code_from_views(
    code_views: tuple[AbstractArray, ...],
    bounds: AbstractArray,
    n_samples: int,
    n_components: int,
) -> AbstractArray:
    """Describe the assembled dense sparse_encode code matrix."""
    del code_views, bounds
    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    if n_components < 1:
        raise ValueError("n_components must be positive")
    return AbstractArray(shape=(n_samples, n_components), dtype="float64")
