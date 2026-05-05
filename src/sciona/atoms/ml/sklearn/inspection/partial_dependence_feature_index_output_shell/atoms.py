"""Partial-dependence feature-index output shell atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_partial_dependence_feature_indices_array,
    witness_partial_dependence_feature_indices_vector,
    witness_partial_dependence_selected_feature_count,
)


def _column_index_tuple(value: object) -> bool:
    return bool(
        isinstance(value, tuple)
        and len(value) >= 1
        and all(isinstance(item, int) and not isinstance(item, bool) for item in value)
    )


def _feature_index_array(value: object) -> bool:
    if not isinstance(value, np.ndarray):
        return False
    return bool(value.size >= 1 and np.issubdtype(value.dtype, np.integer))


def _nonnegative_int(value: object) -> bool:
    return bool(isinstance(value, int) and not isinstance(value, bool) and value >= 0)


@register_atom(witness_partial_dependence_feature_indices_array)
@icontract.require(lambda column_indices: _column_index_tuple(column_indices), "column_indices must be a nonempty tuple of ints")
@icontract.ensure(
    lambda result, column_indices: isinstance(result, np.ndarray)
    and result.dtype == np.dtype(np.intp)
    and result.flags["C_CONTIGUOUS"]
    and np.array_equal(result, np.asarray(column_indices, dtype=np.intp, order="C")),
    "result must be sklearn's C-order np.intp array of column indices",
)
def partial_dependence_feature_indices_array(
    column_indices: tuple[int, ...],
) -> NDArray[np.intp]:
    """Convert selected column indices to sklearn's C-order np.intp array."""
    return np.asarray(column_indices, dtype=np.intp, order="C")


@register_atom(witness_partial_dependence_feature_indices_vector)
@icontract.require(lambda feature_indices_array: _feature_index_array(feature_indices_array), "feature_indices_array must be a nonempty integer array")
@icontract.ensure(
    lambda result, feature_indices_array: isinstance(result, np.ndarray)
    and result.dtype == np.dtype(np.intp)
    and result.ndim == 1
    and np.array_equal(result, np.asarray(feature_indices_array, dtype=np.intp).ravel()),
    "result must be sklearn's flattened 1D np.intp feature-index vector",
)
def partial_dependence_feature_indices_vector(
    feature_indices_array: NDArray[np.intp],
) -> NDArray[np.intp]:
    """Flatten sklearn's feature-index array shell to a 1D np.intp vector."""
    return np.asarray(feature_indices_array, dtype=np.intp).ravel()


@register_atom(witness_partial_dependence_selected_feature_count)
@icontract.require(lambda feature_indices: _feature_index_array(feature_indices), "feature_indices must be a nonempty integer array")
@icontract.ensure(lambda result: _nonnegative_int(result), "result must be a nonnegative integer")
def partial_dependence_selected_feature_count(
    feature_indices: NDArray[np.intp],
) -> int:
    """Count the selected feature indices after sklearn's vectorization shell."""
    return int(np.asarray(feature_indices, dtype=np.intp).shape[0])
