"""Witnesses for sklearn SparseCoder inverse shell helpers."""

from __future__ import annotations

import numpy as np

from sciona.ghost.abstract import AbstractArray


def _check_matrix(values: AbstractArray, name: str) -> tuple[int, int]:
    if len(values.shape) != 2:
        raise ValueError(f"{name} must be 2D")
    n_rows = int(values.shape[0])
    n_cols = int(values.shape[1])
    if n_rows < 1 or n_cols < 1:
        raise ValueError(f"{name} must be nonempty")
    return n_rows, n_cols


def witness_sparse_coder_fit_require_matching_features(
    x_feature_count: int,
    dictionary_feature_count: int,
) -> int:
    """Describe SparseCoder.fit feature-count validation."""
    if x_feature_count < 1 or dictionary_feature_count < 1:
        raise ValueError("feature counts must be positive")
    return int(x_feature_count)


def witness_sparse_coding_expected_code_width(
    dictionary: AbstractArray,
    *,
    split_sign: bool = False,
) -> int:
    """Describe expected code width from dictionary atoms and split_sign."""
    n_components, _ = _check_matrix(dictionary, "dictionary")
    return 2 * n_components if split_sign else n_components


def witness_sparse_coding_merge_split_sign(
    code: AbstractArray,
) -> AbstractArray:
    """Describe merging split-sign code blocks back to signed coefficients."""
    n_samples, n_features = _check_matrix(code, "code")
    if n_features % 2 != 0:
        raise ValueError("code width must be even")
    return AbstractArray(shape=(n_samples, n_features // 2), dtype=np.float64)


def witness_sparse_coder_inverse_transform(
    code: AbstractArray,
    dictionary: AbstractArray,
    *,
    split_sign: bool = False,
) -> AbstractArray:
    """Describe SparseCoder inverse-transform output shape."""
    n_samples, code_width = _check_matrix(code, "code")
    n_components, n_features = _check_matrix(dictionary, "dictionary")
    expected_width = 2 * n_components if split_sign else n_components
    if code_width != expected_width:
        raise ValueError("code width must match dictionary row count and split_sign")
    return AbstractArray(shape=(n_samples, n_features), dtype=np.float64)
