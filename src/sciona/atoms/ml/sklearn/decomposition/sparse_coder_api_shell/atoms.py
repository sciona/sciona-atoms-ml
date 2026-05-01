"""SparseCoder API-shell atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_sparse_coder_fit_return_self,
    witness_sparse_coder_n_features_out,
    witness_sparse_coder_preserves_dtype_tags,
    witness_sparse_coder_requires_fit_tag,
    witness_sparse_coder_transform_dictionary,
)

Matrix = NDArray[np.float64]


def _finite_matrix(values: object) -> bool:
    try:
        matrix = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(
        matrix.ndim == 2
        and matrix.shape[0] >= 1
        and matrix.shape[1] >= 1
        and np.all(np.isfinite(matrix))
    )


def _positive_int(value: object) -> bool:
    return bool(isinstance(value, int) and not isinstance(value, bool) and value >= 1)


def _nonempty_string(value: object) -> bool:
    return bool(isinstance(value, str) and value != "")


def _string_tuple(values: object) -> bool:
    return bool(
        isinstance(values, tuple)
        and len(values) >= 1
        and all(isinstance(item, str) and item != "" for item in values)
    )


@register_atom(witness_sparse_coder_fit_return_self)
@icontract.require(lambda estimator_token: _nonempty_string(estimator_token), "estimator_token must be a nonempty string")
@icontract.ensure(lambda result, estimator_token: isinstance(result, str) and result == estimator_token, "result must return the estimator token unchanged")
def sparse_coder_fit_return_self(estimator_token: str) -> str:
    """Model SparseCoder.fit returning the estimator unchanged."""
    return estimator_token


@register_atom(witness_sparse_coder_transform_dictionary)
@icontract.require(lambda dictionary: _finite_matrix(dictionary), "dictionary must be a finite nonempty 2D matrix")
@icontract.ensure(
    lambda result, dictionary: _finite_matrix(result)
    and np.array_equal(np.asarray(result, dtype=np.float64), np.asarray(dictionary, dtype=np.float64)),
    "result must expose the dictionary passed into _transform unchanged",
)
def sparse_coder_transform_dictionary(dictionary: Matrix) -> Matrix:
    """Expose the dictionary that SparseCoder.transform passes to _transform."""
    return np.asarray(dictionary, dtype=np.float64)


@register_atom(witness_sparse_coder_requires_fit_tag)
@icontract.ensure(lambda result: isinstance(result, bool) and result is False, "SparseCoder.requires_fit must be False")
def sparse_coder_requires_fit_tag(parent_requires_fit: bool) -> bool:
    """Override the requires_fit tag for SparseCoder."""
    del parent_requires_fit
    return False


@register_atom(witness_sparse_coder_preserves_dtype_tags)
@icontract.require(lambda parent_preserves_dtype: _string_tuple(parent_preserves_dtype), "parent_preserves_dtype must be a tuple of nonempty strings")
@icontract.ensure(
    lambda result: isinstance(result, tuple) and result == ("float64", "float32"),
    "SparseCoder preserves_dtype tags must be float64 and float32",
)
def sparse_coder_preserves_dtype_tags(parent_preserves_dtype: tuple[str, ...]) -> tuple[str, str]:
    """Override the preserves_dtype transformer tags for SparseCoder."""
    del parent_preserves_dtype
    return ("float64", "float32")


@register_atom(witness_sparse_coder_n_features_out)
@icontract.require(lambda dictionary: _finite_matrix(dictionary), "dictionary must be a finite nonempty 2D matrix")
@icontract.ensure(lambda result: _positive_int(result), "_n_features_out must be a positive integer")
def sparse_coder_n_features_out(dictionary: Matrix) -> int:
    """Expose SparseCoder._n_features_out from the dictionary row count."""
    return int(np.asarray(dictionary, dtype=np.float64).shape[0])
