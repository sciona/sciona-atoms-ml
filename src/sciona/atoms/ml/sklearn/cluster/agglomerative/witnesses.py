"""Ghost witnesses for agglomerative hierarchy cut atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def _check_children(children: AbstractArray, n_leaves: int) -> None:
    if n_leaves < 1:
        raise ValueError("n_leaves must be positive")
    if len(children.shape) != 2:
        raise ValueError("children must be 2D")
    if int(children.shape[0]) != n_leaves - 1 or int(children.shape[1]) != 2:
        raise ValueError("children shape must be (n_leaves - 1, 2)")


def witness_agglomerative_root_node(children: AbstractArray, n_leaves: int) -> int:
    """Describe the root node id of a binary agglomerative hierarchy."""
    _check_children(children, n_leaves)
    return 0


def witness_agglomerative_descendent_leaves(
    node: int,
    children: AbstractArray,
    n_leaves: int,
) -> AbstractArray:
    """Describe the leaf descendants of a hierarchy node."""
    _check_children(children, n_leaves)
    if node < 0 or node > 2 * n_leaves - 2:
        raise ValueError("node must be valid for the hierarchy")
    return AbstractArray(shape=(n_leaves,), dtype="int64")


def witness_agglomerative_hc_cut(
    n_clusters: int,
    children: AbstractArray,
    n_leaves: int,
) -> AbstractArray:
    """Describe cluster labels from cutting a hierarchy."""
    _check_children(children, n_leaves)
    if n_clusters < 1 or n_clusters > n_leaves:
        raise ValueError("n_clusters must fit the leaf count")
    return AbstractArray(shape=(n_leaves,), dtype="int64")
