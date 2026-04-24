"""Ghost witnesses for agglomerative fit preflight atoms."""

from __future__ import annotations


def witness_agglomerative_fit_require_exactly_one_cluster_spec(
    n_clusters: int | None,
    *,
    distance_threshold: float | None = None,
) -> bool:
    """Describe sklearn's exactly-one-cluster-spec preflight guard."""
    del n_clusters, distance_threshold
    return True


def witness_agglomerative_fit_require_full_tree_when_distance_threshold_set(
    compute_full_tree: str | bool,
    *,
    distance_threshold: float | None = None,
) -> bool:
    """Describe sklearn's distance-threshold/full-tree compatibility guard."""
    del compute_full_tree, distance_threshold
    return True


def witness_agglomerative_fit_require_ward_metric_euclidean(
    linkage: str,
    metric: object,
) -> bool:
    """Describe sklearn's Ward-metric compatibility guard."""
    del linkage, metric
    return True
