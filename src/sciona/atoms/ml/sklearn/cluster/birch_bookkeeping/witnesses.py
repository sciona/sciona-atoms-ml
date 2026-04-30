"""Ghost witnesses for sklearn Birch bookkeeping helpers."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_birch_first_call(partial: bool, has_root: bool) -> bool:
    """Describe whether Birch._fit starts a fresh tree."""
    del partial
    del has_root
    return True


def witness_birch_copy_warning_required(copy: str | bool, first_call: bool) -> bool:
    """Describe Birch's copy-deprecation warning predicate."""
    del copy
    del first_call
    return False


def witness_birch_compute_labels_required(has_input_data: bool, compute_labels: bool) -> bool:
    """Describe whether Birch._global_clustering should refresh labels_."""
    del has_input_data
    del compute_labels
    return False


def witness_birch_not_enough_centroids(n_centroids: int, n_clusters: int) -> bool:
    """Describe Birch's no-global-clustering short circuit for insufficient centroids."""
    del n_centroids
    del n_clusters
    return False


def witness_birch_identity_subcluster_labels(n_centroids: int) -> AbstractArray:
    """Describe Birch's identity subcluster labels in the no-global-clustering branch."""
    if n_centroids < 1:
        raise ValueError("n_centroids must be positive")
    return AbstractArray(shape=(n_centroids,), dtype="int64")


def witness_birch_leaf_centers(leaf_centroid_blocks: tuple[AbstractArray, ...]) -> AbstractArray:
    """Describe concatenated Birch leaf-centroid blocks."""
    if len(leaf_centroid_blocks) < 1:
        raise ValueError("leaf_centroid_blocks must be nonempty")
    n_features = int(leaf_centroid_blocks[0].shape[1])
    total_rows = 0
    for block in leaf_centroid_blocks:
        if len(block.shape) != 2:
            raise ValueError("each centroid block must be 2D")
        if int(block.shape[0]) < 1:
            raise ValueError("each centroid block must contain at least one row")
        if int(block.shape[1]) != n_features:
            raise ValueError("all centroid blocks must share a feature count")
        total_rows += int(block.shape[0])
    return AbstractArray(shape=(total_rows, n_features), dtype="float64")


def witness_birch_n_features_out(subcluster_centers: AbstractArray) -> int:
    """Describe Birch's transformed output width from subcluster centers."""
    if len(subcluster_centers.shape) != 2:
        raise ValueError("subcluster_centers must be 2D")
    if int(subcluster_centers.shape[0]) < 1 or int(subcluster_centers.shape[1]) < 1:
        raise ValueError("subcluster_centers must be nonempty")
    return int(subcluster_centers.shape[0])
