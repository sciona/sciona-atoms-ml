"""Ghost witnesses for spectral coclustering preflight atoms."""

from __future__ import annotations


def witness_cocluster_checked_n_clusters(n_clusters: int, n_samples: int) -> int:
    """Describe the validated spectral coclustering cluster count."""
    del n_samples
    return n_clusters
