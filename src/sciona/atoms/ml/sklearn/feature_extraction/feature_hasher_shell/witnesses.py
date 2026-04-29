"""Witnesses for sklearn FeatureHasher shell helpers."""

from __future__ import annotations

from scipy.sparse import csr_matrix

from sciona.ghost.abstract import AbstractArray


def witness_feature_hasher_dict_items(
    raw_samples: AbstractArray,
) -> AbstractArray:
    """Describe dict-mode normalization to per-sample key-value tuples."""
    if len(raw_samples.shape) != 1:
        raise ValueError("raw_samples must be 1D")
    return AbstractArray(shape=raw_samples.shape, dtype="object")


def witness_feature_hasher_pair_items(
    raw_samples: AbstractArray,
) -> AbstractArray:
    """Describe pair-mode normalization to float-valued pair tuples."""
    if len(raw_samples.shape) != 1:
        raise ValueError("raw_samples must be 1D")
    return AbstractArray(shape=raw_samples.shape, dtype="object")


def witness_feature_hasher_string_items(
    raw_samples: AbstractArray,
) -> AbstractArray:
    """Describe string-mode normalization to unit-weight pair tuples."""
    if len(raw_samples.shape) != 1:
        raise ValueError("raw_samples must be 1D")
    return AbstractArray(shape=raw_samples.shape, dtype="object")


def witness_feature_hasher_sample_count(
    indptr: AbstractArray,
) -> int:
    """Describe sample-count extraction from a CSR indptr vector."""
    if len(indptr.shape) != 1:
        raise ValueError("indptr must be 1D")
    if int(indptr.shape[0]) < 1:
        raise ValueError("indptr must be nonempty")
    return int(indptr.shape[0]) - 1


def witness_feature_hasher_require_nonempty_samples(
    n_samples: int,
) -> int:
    """Describe the positive sample-count guard."""
    if n_samples < 0:
        raise ValueError("n_samples must be nonnegative")
    return max(1, n_samples)


def witness_feature_hasher_csr_matrix(
    indices: AbstractArray,
    indptr: AbstractArray,
    values: AbstractArray,
    *,
    n_features: int,
    dtype_name: str = "float64",
) -> csr_matrix:
    """Describe the CSR output matrix built from hashed payload arrays."""
    del dtype_name
    if len(indices.shape) != 1 or len(values.shape) != 1:
        raise ValueError("indices and values must be 1D")
    if len(indptr.shape) != 1:
        raise ValueError("indptr must be 1D")
    if int(indices.shape[0]) != int(values.shape[0]):
        raise ValueError("indices and values must have the same length")
    if int(indptr.shape[0]) < 2:
        raise ValueError("indptr must encode at least one sample")
    if n_features < 1:
        raise ValueError("n_features must be positive")
    n_samples = int(indptr.shape[0]) - 1
    return csr_matrix((n_samples, n_features), dtype=float)
