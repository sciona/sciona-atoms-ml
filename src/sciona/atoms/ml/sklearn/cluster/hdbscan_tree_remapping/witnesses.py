"""Ghost witnesses for HDBSCAN linkage-tree remapping helper atoms."""

from __future__ import annotations

from numpy.typing import NDArray


def witness_hdbscan_tree_node_id(
    node_id: int,
    finite_count: int,
    outlier_count: int,
    internal_to_raw: dict[int, int],
) -> int:
    """Describe HDBSCAN's remapped linkage-tree node id for one endpoint."""
    del node_id
    del finite_count
    del outlier_count
    del internal_to_raw
    return 0


def witness_hdbscan_remapped_tree_rows(
    tree: NDArray[object],
    internal_to_raw: dict[int, int],
    outlier_count: int,
) -> NDArray[object]:
    """Describe HDBSCAN's linkage-tree rows after remapping endpoint ids and before appending outlier rows."""
    del tree
    del internal_to_raw
    del outlier_count
    raise NotImplementedError


def witness_hdbscan_outlier_linkage_rows(
    last_cluster_id: int,
    last_cluster_size: int,
    nonfinite_raw_indices: NDArray[int],
) -> NDArray[object]:
    """Describe HDBSCAN's appended linkage-tree rows for non-finite raw samples."""
    del last_cluster_id
    del last_cluster_size
    del nonfinite_raw_indices
    raise NotImplementedError


def witness_hdbscan_remapped_single_linkage_tree(
    tree: NDArray[object],
    internal_to_raw: dict[int, int],
    nonfinite_raw_indices: NDArray[int],
) -> NDArray[object]:
    """Describe HDBSCAN's final remapped linkage tree after restoring non-finite samples."""
    del tree
    del internal_to_raw
    del nonfinite_raw_indices
    raise NotImplementedError
