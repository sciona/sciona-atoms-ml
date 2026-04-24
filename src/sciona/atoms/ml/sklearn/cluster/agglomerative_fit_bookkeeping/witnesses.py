"""Ghost witnesses for agglomerative fit-bookkeeping atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_agglomerative_resolve_compute_full_tree(
    compute_full_tree: str | bool,
    has_connectivity: bool,
    n_clusters: int,
    n_samples: int,
    distance_threshold: float | None = None,
) -> bool:
    """Describe a resolved agglomerative compute_full_tree flag."""
    del compute_full_tree, has_connectivity, n_clusters, n_samples, distance_threshold
    return True


def witness_agglomerative_resolve_tree_n_clusters(
    n_clusters: int,
    compute_full_tree: bool,
) -> int | None:
    """Describe the tree-builder n_clusters argument after compute_full_tree resolution."""
    del n_clusters, compute_full_tree
    return 0


def witness_agglomerative_return_distance_required(
    distance_threshold: float | None = None,
    compute_distances: bool = False,
) -> bool:
    """Describe whether agglomerative tree construction must return distances."""
    del distance_threshold, compute_distances
    return True


def witness_agglomerative_cluster_count_from_distances(
    distances: AbstractArray,
    *,
    distance_threshold: float,
) -> int:
    """Describe cluster-count derivation from agglomerative merge distances."""
    del distance_threshold
    if len(distances.shape) != 1 or int(distances.shape[0]) < 1:
        raise ValueError("distances must be a nonempty 1D array")
    return 1


def witness_agglomerative_labels_from_heads(
    heads: AbstractArray,
    *,
    n_samples: int,
) -> AbstractArray:
    """Describe relabeled agglomerative heads for the sample prefix."""
    if len(heads.shape) != 1 or int(heads.shape[0]) < n_samples:
        raise ValueError("heads must be 1D and cover at least n_samples entries")
    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    return AbstractArray(shape=(n_samples,), dtype="int64", min_val=0)
