"""Ghost witnesses for sklearn forest apply and decision-path output helpers."""

from __future__ import annotations

from scipy.sparse import csr_matrix

from sciona.ghost.abstract import AbstractArray


def witness_forest_apply_leaf_matrix(
    leaf_vectors: tuple[AbstractArray, ...],
) -> AbstractArray:
    """Describe the sample-major leaf-index matrix returned by BaseForest.apply."""
    if len(leaf_vectors) < 1:
        raise ValueError("leaf_vectors must be nonempty")
    n_estimators = len(leaf_vectors)
    n_samples = int(leaf_vectors[0].shape[0])
    if n_samples < 1:
        raise ValueError("leaf vectors must be nonempty")
    for vector in leaf_vectors:
        if len(vector.shape) != 1 or int(vector.shape[0]) != n_samples:
            raise ValueError("leaf vectors must be aligned 1D arrays")
    return AbstractArray(shape=(n_samples, n_estimators), dtype="int64")


def witness_forest_decision_path_node_ptr(
    indicators: tuple[AbstractArray, ...],
) -> AbstractArray:
    """Describe the cumulative tree-column offsets returned by BaseForest.decision_path."""
    if len(indicators) < 1:
        raise ValueError("indicators must be nonempty")
    n_samples = int(indicators[0].shape[0])
    for block in indicators:
        if len(block.shape) != 2 or int(block.shape[0]) != n_samples:
            raise ValueError("indicator blocks must be aligned 2D matrices")
    return AbstractArray(shape=(len(indicators) + 1,), dtype="int64")


def witness_forest_decision_path_csr(
    indicators: tuple[AbstractArray, ...],
) -> csr_matrix:
    """Describe the CSR decision-path indicator matrix returned by BaseForest.decision_path."""
    if len(indicators) < 1:
        raise ValueError("indicators must be nonempty")
    n_samples = int(indicators[0].shape[0])
    total_columns = 0
    for block in indicators:
        if len(block.shape) != 2 or int(block.shape[0]) != n_samples:
            raise ValueError("indicator blocks must be aligned 2D matrices")
        total_columns += int(block.shape[1])
    return csr_matrix((n_samples, total_columns), dtype=int)
