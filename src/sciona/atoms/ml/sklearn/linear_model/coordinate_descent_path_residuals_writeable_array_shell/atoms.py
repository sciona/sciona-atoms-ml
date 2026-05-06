"""Sklearn coordinate-descent path-residual writeable-array atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_cd_path_residuals_array_needs_writeable_fix,
    witness_cd_path_residuals_dense_writeable_guard,
    witness_cd_path_residuals_writable_array,
)


def _bool(value: object) -> bool:
    return isinstance(value, bool)


def _array_valid(value: object) -> bool:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        return False
    return bool(
        array.ndim in {1, 2}
        and np.issubdtype(array.dtype, np.number)
        and np.all(np.isfinite(array))
    )


def _writable_array_result_valid(result: object, array: object, array_input: object) -> bool:
    if not _array_valid(result) or not _array_valid(array):
        return False
    result_values = np.asarray(result)
    source_values = np.asarray(array)
    needs_fix = bool(array.base is not array_input and not array.flags["WRITEABLE"])
    return bool(
        result is array
        and result_values.shape == source_values.shape
        and np.array_equal(result_values, source_values)
        and (result.flags["WRITEABLE"] if needs_fix else True)
    )


@register_atom(witness_cd_path_residuals_dense_writeable_guard)
@icontract.require(lambda X_is_sparse: _bool(X_is_sparse), "X_is_sparse must be boolean")
@icontract.ensure(
    lambda result, X_is_sparse: _bool(result) and result == (not X_is_sparse),
    "dense writeable-array guard must match not sparse.issparse(X)",
)
def cd_path_residuals_dense_writeable_guard(X_is_sparse: bool) -> bool:
    """Return whether _path_residuals enters the dense writeable-array guard loop."""
    return not X_is_sparse


@register_atom(witness_cd_path_residuals_array_needs_writeable_fix)
@icontract.require(
    lambda array_base_matches_input: _bool(array_base_matches_input),
    "array_base_matches_input must be boolean",
)
@icontract.require(lambda array_writeable: _bool(array_writeable), "array_writeable must be boolean")
@icontract.ensure(
    lambda result, array_base_matches_input, array_writeable: _bool(result)
    and result == ((not array_base_matches_input) and (not array_writeable)),
    "per-array writeability guard must match sklearn branching",
)
def cd_path_residuals_array_needs_writeable_fix(
    array_base_matches_input: bool, array_writeable: bool
) -> bool:
    """Return whether _path_residuals calls array.setflags(write=True) for one array."""
    return (not array_base_matches_input) and (not array_writeable)


@register_atom(witness_cd_path_residuals_writable_array)
@icontract.require(lambda array: _array_valid(array), "array must be a finite numeric rank-1 or rank-2 array")
@icontract.ensure(
    lambda result, array, array_input: _writable_array_result_valid(result, array, array_input),
    "result must preserve the array and enable writeability when sklearn's guard requires it",
)
def cd_path_residuals_writable_array(
    array: NDArray[np.generic], array_input: object
) -> NDArray[np.generic]:
    """Return the array after sklearn's conditional setflags(write=True) normalization."""
    if array.base is not array_input and not array.flags["WRITEABLE"]:
        array.setflags(write=True)
    return array
