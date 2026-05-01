"""Ghost witnesses for spectral biclustering preflight atoms."""

from __future__ import annotations


def witness_bicluster_checked_cluster_counts(
    n_clusters: int | tuple[int, int],
    n_samples: int,
) -> tuple[int, int]:
    """Describe the validated spectral biclustering row and column cluster counts."""
    del n_clusters
    del n_samples
    return 1, 1


def witness_bicluster_checked_n_best(n_best: int, n_components: int) -> int:
    """Describe the validated spectral biclustering n_best value."""
    del n_components
    return n_best


def witness_bicluster_checked_method(method: str, is_sparse: bool) -> str:
    """Describe the validated spectral biclustering normalization method."""
    del is_sparse
    return method
