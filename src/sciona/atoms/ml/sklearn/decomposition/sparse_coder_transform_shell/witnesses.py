"""Witnesses for sklearn sparse-coder transform shell helpers."""

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


def witness_sparse_coding_transform_alpha(
    transform_alpha: float | None,
    *,
    fit_alpha: float | None = None,
) -> float | None:
    """Describe transform-alpha resolution from explicit and fit-time values."""
    if fit_alpha is not None and transform_alpha is None:
        return float(fit_alpha)
    return None if transform_alpha is None else float(transform_alpha)


def witness_sparse_coding_split_sign(
    code: AbstractArray,
) -> AbstractArray:
    """Describe positive/negative block expansion of a code matrix."""
    n_samples, n_features = _check_matrix(code, "code")
    return AbstractArray(shape=(n_samples, 2 * n_features), dtype=np.float64)


def witness_sparse_coder_n_components(dictionary: AbstractArray) -> int:
    """Describe the dictionary row-count property."""
    n_components, _ = _check_matrix(dictionary, "dictionary")
    return n_components


def witness_sparse_coder_n_features_in(dictionary: AbstractArray) -> int:
    """Describe the dictionary feature-count property."""
    _, n_features = _check_matrix(dictionary, "dictionary")
    return n_features
