"""Sklearn tree partial-dependence recursion atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_tree_partial_dependence_averaged_predictions,
    witness_tree_partial_dependence_grid,
    witness_tree_partial_dependence_result,
    witness_tree_partial_dependence_target_features,
)


def _float_grid(values: object) -> bool:
    try:
        array = np.asarray(values, dtype=np.float32)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 2 and array.shape[0] >= 1 and array.shape[1] >= 1 and np.all(np.isfinite(array)))


def _int_vector(values: object) -> bool:
    array = np.asarray(values)
    return bool(array.ndim == 1 and array.shape[0] >= 1 and np.issubdtype(array.dtype, np.integer))


def _prediction_vector(values: object) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 1 and array.shape[0] >= 1 and np.all(np.isfinite(array)))


@register_atom(witness_tree_partial_dependence_grid)
@icontract.require(lambda grid: _float_grid(grid), "grid must be a nonempty finite 2D numeric array")
@icontract.ensure(
    lambda result, grid: _float_grid(result)
    and np.asarray(result).dtype == np.float32
    and np.asarray(result).flags.c_contiguous
    and np.allclose(np.asarray(result, dtype=np.float32), np.asarray(grid, dtype=np.float32)),
    "grid must be normalized to a C-order float32 array",
)
def tree_partial_dependence_grid(
    grid: NDArray[np.floating],
) -> NDArray[np.float32]:
    """Return the normalized grid passed to compute_partial_dependence."""
    return np.asarray(grid, dtype=np.float32, order="C")


@register_atom(witness_tree_partial_dependence_averaged_predictions)
@icontract.require(lambda grid: _float_grid(grid), "grid must be a nonempty finite 2D numeric array")
@icontract.ensure(
    lambda result, grid: _prediction_vector(result)
    and np.asarray(result).dtype == np.float64
    and np.asarray(result).flags.c_contiguous
    and np.asarray(result).shape == (np.asarray(grid).shape[0],)
    and np.allclose(np.asarray(result, dtype=np.float64), 0.0),
    "averaged_predictions must be a zero float64 vector sized to grid rows",
)
def tree_partial_dependence_averaged_predictions(
    grid: NDArray[np.floating],
) -> NDArray[np.float64]:
    """Return the zeroed averaged_predictions buffer."""
    array = np.asarray(grid, dtype=np.float32, order="C")
    return np.zeros(shape=array.shape[0], dtype=np.float64, order="C")


@register_atom(witness_tree_partial_dependence_target_features)
@icontract.require(
    lambda target_features: _int_vector(target_features),
    "target_features must be a nonempty one-dimensional integer vector",
)
@icontract.ensure(
    lambda result, target_features: _int_vector(result)
    and np.asarray(result).dtype == np.intp
    and np.asarray(result).flags.c_contiguous
    and np.array_equal(np.asarray(result, dtype=np.intp), np.asarray(target_features, dtype=np.intp)),
    "target_features must be normalized to a C-order intp vector",
)
def tree_partial_dependence_target_features(
    target_features: NDArray[np.integer],
) -> NDArray[np.intp]:
    """Return the normalized target_features vector."""
    return np.asarray(target_features, dtype=np.intp, order="C")


@register_atom(witness_tree_partial_dependence_result)
@icontract.require(
    lambda averaged_predictions: _prediction_vector(averaged_predictions),
    "averaged_predictions must be a nonempty finite 1D float vector",
)
@icontract.ensure(
    lambda result, averaged_predictions: _prediction_vector(result)
    and np.allclose(np.asarray(result, dtype=np.float64), np.asarray(averaged_predictions, dtype=np.float64)),
    "partial-dependence recursion must return the supplied averaged_predictions vector",
)
def tree_partial_dependence_result(
    averaged_predictions: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return the final averaged_predictions vector."""
    return np.asarray(averaged_predictions, dtype=np.float64)
