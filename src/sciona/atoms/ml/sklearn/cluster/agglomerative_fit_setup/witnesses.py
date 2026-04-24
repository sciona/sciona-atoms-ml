"""Ghost witnesses for agglomerative fit setup atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_agglomerative_fit_select_tree_builder(linkage: str) -> str:
    """Describe agglomerative tree-builder lookup from linkage."""
    if linkage not in {"ward", "complete", "average", "single"}:
        raise ValueError("linkage must be one of ward, complete, average, or single")
    return "tree_builder"


def witness_agglomerative_fit_prepare_connectivity(
    X: AbstractArray,
    connectivity: object | None,
) -> AbstractArray | None:
    """Describe agglomerative connectivity preparation before fit bookkeeping."""
    if len(X.shape) != 2:
        raise ValueError("X must be 2D")
    if int(X.shape[0]) < 1 or int(X.shape[1]) < 1:
        raise ValueError("X must be nonempty")
    if connectivity is None:
        return None
    if not isinstance(connectivity, AbstractArray) or len(connectivity.shape) != 2:
        raise ValueError("connectivity must be a 2D matrix when provided")
    rows, cols = int(connectivity.shape[0]), int(connectivity.shape[1])
    if rows != cols or rows != int(X.shape[0]):
        raise ValueError("connectivity must be square and match the sample count in X")
    return AbstractArray(shape=(rows, cols), dtype="float64")
