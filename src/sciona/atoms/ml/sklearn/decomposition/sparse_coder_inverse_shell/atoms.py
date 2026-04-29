"""SparseCoder fit and inverse-transform shell atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_sparse_coder_fit_require_matching_features,
    witness_sparse_coder_inverse_transform,
    witness_sparse_coding_expected_code_width,
    witness_sparse_coding_merge_split_sign,
)

Matrix = NDArray[np.float64]


def _finite_matrix(values: object) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(
        array.ndim == 2
        and array.shape[0] >= 1
        and array.shape[1] >= 1
        and np.all(np.isfinite(array))
    )


def _flag_valid(value: object) -> bool:
    return isinstance(value, bool)


def _matching_feature_counts_valid(
    x_feature_count: object,
    dictionary_feature_count: object,
) -> bool:
    return bool(
        isinstance(x_feature_count, int)
        and not isinstance(x_feature_count, bool)
        and x_feature_count >= 1
        and isinstance(dictionary_feature_count, int)
        and not isinstance(dictionary_feature_count, bool)
        and dictionary_feature_count >= 1
    )


def _positive_int(value: object) -> bool:
    return bool(isinstance(value, int) and not isinstance(value, bool) and value >= 1)


def _expected_width_valid(result: int, dictionary: Matrix, split_sign: bool) -> bool:
    dictionary_values = np.asarray(dictionary, dtype=np.float64)
    expected = int(dictionary_values.shape[0]) * (2 if split_sign else 1)
    return bool(isinstance(result, int) and result == expected)


def _merge_inputs_valid(code: Matrix) -> bool:
    values = np.asarray(code, dtype=np.float64)
    return bool(_finite_matrix(values) and values.shape[1] % 2 == 0)


def _merged_code_valid(result: Matrix, code: Matrix) -> bool:
    result_values = np.asarray(result, dtype=np.float64)
    code_values = np.asarray(code, dtype=np.float64)
    half_width = code_values.shape[1] // 2
    return bool(
        result_values.shape == (code_values.shape[0], half_width)
        and np.all(np.isfinite(result_values))
        and np.allclose(result_values, code_values[:, :half_width] - code_values[:, half_width:])
    )


def _inverse_transform_inputs_valid(
    code: Matrix,
    dictionary: Matrix,
    split_sign: bool,
) -> bool:
    code_values = np.asarray(code, dtype=np.float64)
    dictionary_values = np.asarray(dictionary, dtype=np.float64)
    expected_width = dictionary_values.shape[0] * (2 if split_sign else 1)
    return bool(
        _finite_matrix(code_values)
        and _finite_matrix(dictionary_values)
        and code_values.shape[1] == expected_width
    )


def _inverse_transform_valid(result: Matrix, code: Matrix, dictionary: Matrix) -> bool:
    result_values = np.asarray(result, dtype=np.float64)
    code_values = np.asarray(code, dtype=np.float64)
    dictionary_values = np.asarray(dictionary, dtype=np.float64)
    return bool(
        result_values.shape == (code_values.shape[0], dictionary_values.shape[1])
        and np.all(np.isfinite(result_values))
    )


@register_atom(witness_sparse_coder_fit_require_matching_features)
@icontract.require(
    lambda x_feature_count, dictionary_feature_count: _matching_feature_counts_valid(
        x_feature_count, dictionary_feature_count
    ),
    "x_feature_count and dictionary_feature_count must be positive integers",
)
@icontract.ensure(lambda result, x_feature_count: result == int(x_feature_count), "validated feature count must equal X's feature count")
def sparse_coder_fit_require_matching_features(
    x_feature_count: int,
    dictionary_feature_count: int,
) -> int:
    """Require SparseCoder.fit to see the same feature count in X and the dictionary."""
    if int(x_feature_count) != int(dictionary_feature_count):
        raise ValueError(
            "Dictionary and X have different numbers of features:"
            f"dictionary.shape: ({dictionary_feature_count},) X.shape: ({x_feature_count},)"
        )
    return int(x_feature_count)


@register_atom(witness_sparse_coding_expected_code_width)
@icontract.require(lambda dictionary: _finite_matrix(dictionary), "dictionary must be a finite nonempty matrix")
@icontract.require(lambda split_sign=False: _flag_valid(split_sign), "split_sign must be boolean")
@icontract.ensure(lambda result, dictionary, split_sign=False: _expected_width_valid(result, dictionary, split_sign), "expected code width must match the dictionary row count and split_sign setting")
def sparse_coding_expected_code_width(
    dictionary: Matrix,
    *,
    split_sign: bool = False,
) -> int:
    """Compute the code width expected by SparseCoder inverse_transform."""
    dictionary_values = np.asarray(dictionary, dtype=np.float64)
    base_width = int(dictionary_values.shape[0])
    return 2 * base_width if split_sign else base_width


@register_atom(witness_sparse_coding_merge_split_sign)
@icontract.require(lambda code: _merge_inputs_valid(code), "code must be a finite sample-by-feature matrix with even width")
@icontract.ensure(lambda result, code: _merged_code_valid(result, code), "merged code must subtract the negative block from the positive block")
def sparse_coding_merge_split_sign(
    code: Matrix,
) -> Matrix:
    """Merge split-sign code blocks back into signed component coefficients."""
    code_values = np.asarray(code, dtype=np.float64)
    half_width = code_values.shape[1] // 2
    return np.asarray(code_values[:, :half_width] - code_values[:, half_width:], dtype=np.float64)


@register_atom(witness_sparse_coder_inverse_transform)
@icontract.require(lambda code, dictionary, split_sign=False: _inverse_transform_inputs_valid(code, dictionary, split_sign), "code width must match the dictionary row count and split_sign setting")
@icontract.require(lambda split_sign=False: _flag_valid(split_sign), "split_sign must be boolean")
@icontract.ensure(lambda result, code, dictionary: _inverse_transform_valid(result, code, dictionary), "inverse transform must return a finite sample-by-feature reconstruction")
def sparse_coder_inverse_transform(
    code: Matrix,
    dictionary: Matrix,
    *,
    split_sign: bool = False,
) -> Matrix:
    """Reconstruct original-space features from SparseCoder codes and dictionary atoms."""
    code_values = np.asarray(code, dtype=np.float64)
    dictionary_values = np.asarray(dictionary, dtype=np.float64)
    signed_code = sparse_coding_merge_split_sign(code_values) if split_sign else code_values
    return np.asarray(signed_code @ dictionary_values, dtype=np.float64)
