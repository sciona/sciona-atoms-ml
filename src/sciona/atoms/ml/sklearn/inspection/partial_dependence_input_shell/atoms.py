"""Partial-dependence input-shell atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray
from sklearn.utils import check_array

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_partial_dependence_checked_object_array,
    witness_partial_dependence_use_object_check_array,
)


ObjectMatrix = NDArray[np.object_]


def _bool_scalar(value: object) -> bool:
    return isinstance(value, bool)


def _array_like_2d(values: object) -> bool:
    try:
        array = np.asarray(values, dtype=object)
    except Exception:
        return False
    return bool(array.ndim == 2 and array.shape[0] >= 1 and array.shape[1] >= 1)


def _object_matrix(values: object) -> bool:
    try:
        array = np.asarray(values, dtype=object)
    except Exception:
        return False
    return bool(array.ndim == 2 and array.shape[0] >= 1 and array.shape[1] >= 1)


@register_atom(witness_partial_dependence_use_object_check_array)
@icontract.require(lambda has_array: _bool_scalar(has_array), "has_array must be boolean")
@icontract.require(lambda is_sparse: _bool_scalar(is_sparse), "is_sparse must be boolean")
@icontract.ensure(lambda result: isinstance(result, bool), "result must be boolean")
def partial_dependence_use_object_check_array(
    has_array: bool,
    is_sparse: bool,
) -> bool:
    """Decide whether partial_dependence normalizes X with object-dtype check_array."""
    return not (bool(has_array) or bool(is_sparse))


@register_atom(witness_partial_dependence_checked_object_array)
@icontract.require(lambda X: _array_like_2d(X), "X must be array-like 2D")
@icontract.ensure(lambda result: _object_matrix(result), "checked object array must be 2D and object-coercible")
def partial_dependence_checked_object_array(
    X: object,
) -> ObjectMatrix:
    """Apply sklearn's object-dtype check_array branch for non-array-like partial_dependence input."""
    return np.asarray(check_array(X, ensure_all_finite="allow-nan", dtype=object), dtype=object)
