"""Ghost witnesses for sklearn permutation-importance preprocessing helpers."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_permutation_importance_max_sample_count(max_samples: float | int, n_samples: int) -> int:
    """Describe sklearn's effective max-sample count for permutation importance."""
    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    if isinstance(max_samples, int):
        if max_samples < 1 or max_samples > n_samples:
            raise ValueError("integer max_samples must lie in [1, n_samples]")
    elif not (0.0 < float(max_samples) <= 1.0):
        raise ValueError("float max_samples must lie in (0, 1]")
    return 1


def witness_permutation_importance_row_indices(n_population: int, n_samples: int) -> AbstractArray:
    """Describe sklearn's sampled row indices for max_samples subsampling."""
    if n_population < 1:
        raise ValueError("n_population must be positive")
    if n_samples < 1 or n_samples > n_population:
        raise ValueError("n_samples must lie in [1, n_population]")
    return AbstractArray(shape=(n_samples,), dtype="int64")


def witness_permutation_importance_shuffle_indices(n_samples: int, n_repeats: int) -> AbstractArray:
    """Describe repeated in-place shuffle-index states for one feature column."""
    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    if n_repeats < 1:
        raise ValueError("n_repeats must be positive")
    return AbstractArray(shape=(n_repeats, n_samples), dtype="int64")


def witness_permutation_importance_dense_permuted_columns(
    X: AbstractArray,
    col_idx: int,
    shuffle_indices: AbstractArray,
) -> AbstractArray:
    """Describe repeated dense column permutations for one feature."""
    if len(X.shape) != 2:
        raise ValueError("X must be 2D")
    n_samples = int(X.shape[0])
    n_features = int(X.shape[1])
    if n_samples < 1 or n_features < 1:
        raise ValueError("X must be nonempty")
    if col_idx < 0 or col_idx >= n_features:
        raise ValueError("col_idx out of range")
    if len(shuffle_indices.shape) != 2:
        raise ValueError("shuffle_indices must be 2D")
    if int(shuffle_indices.shape[1]) != n_samples:
        raise ValueError("shuffle indices must match X rows")
    return AbstractArray(shape=(int(shuffle_indices.shape[0]), n_samples, n_features), dtype="float64")
