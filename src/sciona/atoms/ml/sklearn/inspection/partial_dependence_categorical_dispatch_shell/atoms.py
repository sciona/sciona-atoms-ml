"""Partial-dependence categorical-dispatch shell atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_partial_dependence_categorical_array,
    witness_partial_dependence_categorical_bool_branch,
    witness_partial_dependence_categorical_index_or_name_branch,
)


def _categorical_input_arrayable(value: object) -> bool:
    try:
        array = np.asarray(value)
    except Exception:
        return False
    return bool(array.ndim >= 1 and array.size >= 1)


def _dtype_kind_valid(value: object) -> bool:
    return isinstance(value, str) and len(value) >= 1


def _bool_or_object_or_string_or_int_array(result: object) -> bool:
    if not isinstance(result, np.ndarray):
        return False
    return bool(result.ndim >= 1 and result.size >= 1)


@register_atom(witness_partial_dependence_categorical_array)
@icontract.require(
    lambda categorical_features: _categorical_input_arrayable(categorical_features),
    "categorical_features must be convertible to a nonempty numpy array",
)
@icontract.ensure(
    lambda result: _bool_or_object_or_string_or_int_array(result),
    "result must be a nonempty numpy array",
)
def partial_dependence_categorical_array(
    categorical_features: object,
) -> NDArray[np.generic]:
    """Coerce sklearn's categorical_features input with np.asarray before dtype branching."""
    return np.asarray(categorical_features)


@register_atom(witness_partial_dependence_categorical_bool_branch)
@icontract.require(lambda dtype_kind: _dtype_kind_valid(dtype_kind), "dtype_kind must be a nonempty string")
@icontract.ensure(lambda result: isinstance(result, bool), "result must be boolean")
def partial_dependence_categorical_bool_branch(
    dtype_kind: str,
) -> bool:
    """Decide whether sklearn takes the boolean-mask categorical branch."""
    return dtype_kind == "b"


@register_atom(witness_partial_dependence_categorical_index_or_name_branch)
@icontract.require(lambda dtype_kind: _dtype_kind_valid(dtype_kind), "dtype_kind must be a nonempty string")
@icontract.ensure(lambda result: isinstance(result, bool), "result must be boolean")
def partial_dependence_categorical_index_or_name_branch(
    dtype_kind: str,
) -> bool:
    """Decide whether sklearn takes the integer-or-name categorical branch."""
    return dtype_kind in ("i", "O", "U")
