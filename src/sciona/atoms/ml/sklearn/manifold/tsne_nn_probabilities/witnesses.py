"""Ghost witnesses for nearest-neighbor t-SNE probability helpers."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_tsne_nn_distance_blocks(distances: AbstractArray) -> AbstractArray:
    """Describe the dense nearest-neighbor distance blocks extracted from CSR storage."""
    if len(distances.shape) != 2:
        raise ValueError("distances must be a matrix")
    n_samples = int(distances.shape[0])
    if n_samples < 2:
        raise ValueError("distances must have at least two samples")
    return AbstractArray(shape=(n_samples, None), dtype="float32")


def witness_tsne_nn_conditional_probability_matrix(
    conditional_probabilities: AbstractArray,
    indices: AbstractArray,
    indptr: AbstractArray,
    *,
    n_samples: int,
) -> AbstractArray:
    """Describe the CSR conditional-probability matrix rebuilt from nearest-neighbor outputs."""
    del indices, indptr
    if len(conditional_probabilities.shape) != 2:
        raise ValueError("conditional_probabilities must be a matrix")
    if n_samples < 2:
        raise ValueError("n_samples must be at least two")
    return AbstractArray(shape=(n_samples, n_samples), dtype="float64")


def witness_tsne_nn_joint_probabilities(
    conditional_probability_matrix: AbstractArray,
) -> AbstractArray:
    """Describe the normalized symmetric CSR joint-probability matrix for nearest-neighbor t-SNE."""
    if len(conditional_probability_matrix.shape) != 2:
        raise ValueError("conditional_probability_matrix must be a matrix")
    n_rows = int(conditional_probability_matrix.shape[0])
    n_cols = int(conditional_probability_matrix.shape[1])
    if n_rows < 2 or n_rows != n_cols:
        raise ValueError("conditional_probability_matrix must be square with at least two samples")
    return AbstractArray(shape=(n_rows, n_cols), dtype="float64")
