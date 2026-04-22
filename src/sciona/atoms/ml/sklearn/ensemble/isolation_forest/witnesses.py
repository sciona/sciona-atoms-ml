"""Ghost witnesses for sklearn IsolationForest helper atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def _check_vector(values: AbstractArray, name: str) -> int:
    if len(values.shape) != 1:
        raise ValueError(f"{name} must be 1D")
    size = int(values.shape[0])
    if size < 1:
        raise ValueError(f"{name} must be nonempty")
    return size


def witness_isolation_forest_average_path_length(n_samples_leaf: AbstractArray) -> AbstractArray:
    """Describe expected isolation-tree path lengths for leaf sample counts."""
    size = _check_vector(n_samples_leaf, "n_samples_leaf")
    return AbstractArray(shape=(size,), dtype="float64")


def witness_isolation_forest_leaf_depths(
    leaf_indices: AbstractArray,
    tree_decision_path_lengths: AbstractArray,
    tree_average_path_lengths: AbstractArray,
) -> AbstractArray:
    """Describe per-sample depth contributions from reached leaves in one tree."""
    size = _check_vector(leaf_indices, "leaf_indices")
    n_nodes = _check_vector(tree_decision_path_lengths, "tree_decision_path_lengths")
    if _check_vector(tree_average_path_lengths, "tree_average_path_lengths") != n_nodes:
        raise ValueError("tree_average_path_lengths must match tree_decision_path_lengths")
    return AbstractArray(shape=(size,), dtype="float64")


def witness_isolation_forest_raw_scores(
    depths: AbstractArray,
    *,
    n_estimators: int,
    max_samples: int,
) -> AbstractArray:
    """Describe raw positive IsolationForest scores from accumulated depths."""
    del n_estimators, max_samples
    size = _check_vector(depths, "depths")
    return AbstractArray(shape=(size,), dtype="float64")
