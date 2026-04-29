"""Sparse-coder transform-shell atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_sparse_coder_n_components,
    witness_sparse_coder_n_features_in,
    witness_sparse_coding_split_sign,
    witness_sparse_coding_transform_alpha,
)

Matrix = NDArray[np.float64]


def _finite_matrix(values: object) -> bool:
    try:
        matrix = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(matrix.ndim == 2 and matrix.shape[0] >= 1 and matrix.shape[1] >= 1 and np.all(np.isfinite(matrix)))


def _optional_nonnegative_float(value: object) -> bool:
    return bool(
        value is None
        or (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and np.isfinite(float(value))
            and float(value) >= 0.0
        )
    )


def _optional_transform_alpha_valid(transform_alpha: object, fit_alpha: object) -> bool:
    return _optional_nonnegative_float(transform_alpha) and _optional_nonnegative_float(fit_alpha)


def _positive_int(value: object) -> bool:
    return bool(isinstance(value, int) and not isinstance(value, bool) and value >= 1)


def _same_shape(result: object, source: object) -> bool:
    try:
        left = np.asarray(result)
        right = np.asarray(source)
    except (TypeError, ValueError):
        return False
    return bool(left.shape == right.shape)


def _split_sign_valid(result: object, code: object) -> bool:
    try:
        result_values = np.asarray(result, dtype=np.float64)
        code_values = np.asarray(code, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    if not (_finite_matrix(code) and result_values.ndim == 2 and np.all(np.isfinite(result_values))):
        return False
    n_samples, n_features = code_values.shape
    if result_values.shape != (n_samples, 2 * n_features):
        return False
    return bool(
        np.allclose(result_values[:, :n_features], np.maximum(code_values, 0.0))
        and np.allclose(result_values[:, n_features:], -np.minimum(code_values, 0.0))
        and np.all(result_values >= 0.0)
    )


@register_atom(witness_sparse_coding_transform_alpha)
@icontract.require(
    lambda transform_alpha, fit_alpha=None: _optional_transform_alpha_valid(transform_alpha, fit_alpha),
    "transform_alpha and fit_alpha must be None or finite nonnegative scalars",
)
@icontract.ensure(
    lambda result: _optional_nonnegative_float(result),
    "resolved transform alpha must be None or a finite nonnegative scalar",
)
def sparse_coding_transform_alpha(
    transform_alpha: float | None,
    *,
    fit_alpha: float | None = None,
) -> float | None:
    """Resolve sklearn's transform alpha, defaulting to fit-time alpha when transform_alpha is unset."""
    if fit_alpha is not None and transform_alpha is None:
        return float(fit_alpha)
    if transform_alpha is None:
        return None
    return float(transform_alpha)


@register_atom(witness_sparse_coding_split_sign)
@icontract.require(lambda code: _finite_matrix(code), "code must be a finite nonempty 2D matrix")
@icontract.ensure(lambda result, code: _split_sign_valid(result, code), "split-sign code must double features into nonnegative positive and negative blocks")
def sparse_coding_split_sign(code: Matrix) -> Matrix:
    """Split each sparse-code feature into concatenated positive and negative parts."""
    values = np.asarray(code, dtype=np.float64)
    n_samples, n_features = values.shape
    split_code = np.empty((n_samples, 2 * n_features), dtype=np.float64)
    split_code[:, :n_features] = np.maximum(values, 0.0)
    split_code[:, n_features:] = -np.minimum(values, 0.0)
    return split_code


@register_atom(witness_sparse_coder_n_components)
@icontract.require(lambda dictionary: _finite_matrix(dictionary), "dictionary must be a finite nonempty 2D matrix")
@icontract.ensure(lambda result: _positive_int(result), "n_components must be a positive integer")
def sparse_coder_n_components(dictionary: Matrix) -> int:
    """Expose sklearn SparseCoder's n_components_ property from the dictionary row count."""
    return int(np.asarray(dictionary, dtype=np.float64).shape[0])


@register_atom(witness_sparse_coder_n_features_in)
@icontract.require(lambda dictionary: _finite_matrix(dictionary), "dictionary must be a finite nonempty 2D matrix")
@icontract.ensure(lambda result: _positive_int(result), "n_features_in must be a positive integer")
def sparse_coder_n_features_in(dictionary: Matrix) -> int:
    """Expose sklearn SparseCoder's n_features_in_ property from the dictionary column count."""
    return int(np.asarray(dictionary, dtype=np.float64).shape[1])
