"""Sklearn coordinate-descent estimator loop-tail atoms."""

from __future__ import annotations

from collections.abc import Sequence

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_cd_estimator_coef_matrix_with_target,
    witness_cd_estimator_dual_gaps_with_target,
    witness_cd_estimator_n_iter_with_target,
    witness_cd_estimator_target_coef_column,
    witness_cd_estimator_target_dual_gap_scalar,
    witness_cd_estimator_target_iteration_count,
)


def _nonnegative_int(value: object) -> bool:
    return isinstance(value, (int, np.integer)) and int(value) >= 0


def _positive_int(value: object) -> bool:
    return isinstance(value, (int, np.integer)) and int(value) >= 1


def _finite_2d_column_source(value: object) -> bool:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        return False
    return bool(
        array.ndim == 2
        and array.shape[0] >= 1
        and array.shape[1] >= 1
        and np.all(np.isfinite(array))
    )


def _finite_vector(value: object) -> bool:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 1 and array.shape[0] >= 1 and np.all(np.isfinite(array)))


def _iteration_source(value: object) -> bool:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 1 and array.shape[0] >= 1 and _positive_int(array[0]))


@register_atom(witness_cd_estimator_target_coef_column)
@icontract.require(
    lambda this_coef: _finite_2d_column_source(this_coef),
    "this_coef must be a finite 2D array with at least one column",
)
@icontract.ensure(
    lambda result, this_coef: isinstance(result, np.ndarray)
    and result.ndim == 1
    and np.array_equal(result, np.asarray(this_coef)[:, 0]),
    "target coefficient vector must equal this_coef[:, 0]",
)
def cd_estimator_target_coef_column(this_coef: object) -> NDArray[np.floating]:
    """Return the per-target coefficient vector from path output."""
    return np.asarray(this_coef)[:, 0]


@register_atom(witness_cd_estimator_coef_matrix_with_target)
@icontract.require(
    lambda coef_matrix: isinstance(coef_matrix, np.ndarray)
    and coef_matrix.ndim == 2
    and coef_matrix.shape[0] >= 1
    and coef_matrix.shape[1] >= 1
    and np.all(np.isfinite(coef_matrix)),
    "coef_matrix must be a finite 2D array",
)
@icontract.require(
    lambda target_index, coef_matrix: _nonnegative_int(target_index)
    and int(target_index) < coef_matrix.shape[0],
    "target_index must select a coefficient row",
)
@icontract.require(
    lambda this_coef, coef_matrix: _finite_2d_column_source(this_coef)
    and np.asarray(this_coef).shape[0] == coef_matrix.shape[1],
    "this_coef[:, 0] must match the coefficient row width",
)
@icontract.ensure(
    lambda result, coef_matrix, target_index, this_coef: isinstance(result, np.ndarray)
    and result is not coef_matrix
    and result.shape == coef_matrix.shape
    and result.dtype == coef_matrix.dtype
    and np.array_equal(result[int(target_index)], np.asarray(this_coef)[:, 0])
    and all(
        np.array_equal(result[row], coef_matrix[row])
        for row in range(coef_matrix.shape[0])
        if row != int(target_index)
    ),
    "updated coefficient matrix must replace only coef_[k] with this_coef[:, 0]",
)
def cd_estimator_coef_matrix_with_target(
    coef_matrix: NDArray[np.floating], target_index: int, this_coef: object
) -> NDArray[np.floating]:
    """Return a coefficient matrix with one target row updated."""
    result = np.array(coef_matrix, copy=True)
    result[int(target_index)] = np.asarray(this_coef)[:, 0]
    return result


@register_atom(witness_cd_estimator_target_dual_gap_scalar)
@icontract.require(lambda this_dual_gap: _finite_vector(this_dual_gap), "this_dual_gap must be a finite vector")
@icontract.ensure(
    lambda result, this_dual_gap: np.isfinite(float(result))
    and np.isclose(float(result), float(np.asarray(this_dual_gap)[0])),
    "target dual gap must equal this_dual_gap[0]",
)
def cd_estimator_target_dual_gap_scalar(this_dual_gap: object) -> float:
    """Return the per-target scalar dual gap from path output."""
    return float(np.asarray(this_dual_gap)[0])


@register_atom(witness_cd_estimator_dual_gaps_with_target)
@icontract.require(lambda dual_gaps: _finite_vector(dual_gaps), "dual_gaps must be a finite vector")
@icontract.require(
    lambda target_index, dual_gaps: _nonnegative_int(target_index)
    and int(target_index) < np.asarray(dual_gaps).shape[0],
    "target_index must select a dual-gap slot",
)
@icontract.require(lambda this_dual_gap: _finite_vector(this_dual_gap), "this_dual_gap must be a finite vector")
@icontract.ensure(
    lambda result, dual_gaps, target_index, this_dual_gap: isinstance(result, np.ndarray)
    and result is not dual_gaps
    and result.shape == np.asarray(dual_gaps).shape
    and result.dtype == np.asarray(dual_gaps).dtype
    and np.isclose(float(result[int(target_index)]), float(np.asarray(this_dual_gap)[0]))
    and all(
        np.isclose(float(result[index]), float(np.asarray(dual_gaps)[index]))
        for index in range(np.asarray(dual_gaps).shape[0])
        if index != int(target_index)
    ),
    "updated dual gaps must replace only dual_gaps_[k] with this_dual_gap[0]",
)
def cd_estimator_dual_gaps_with_target(
    dual_gaps: NDArray[np.floating], target_index: int, this_dual_gap: object
) -> NDArray[np.floating]:
    """Return a dual-gap vector with one target slot updated."""
    result = np.array(dual_gaps, copy=True)
    result[int(target_index)] = np.asarray(this_dual_gap)[0]
    return result


@register_atom(witness_cd_estimator_target_iteration_count)
@icontract.require(lambda this_iter: _iteration_source(this_iter), "this_iter must start with a positive iteration count")
@icontract.ensure(
    lambda result, this_iter: _positive_int(result)
    and int(result) == int(np.asarray(this_iter)[0]),
    "target iteration count must equal this_iter[0]",
)
def cd_estimator_target_iteration_count(this_iter: object) -> int:
    """Return the per-target iteration count from path output."""
    return int(np.asarray(this_iter)[0])


@register_atom(witness_cd_estimator_n_iter_with_target)
@icontract.require(
    lambda n_iter_list: isinstance(n_iter_list, Sequence)
    and all(_positive_int(value) for value in n_iter_list),
    "n_iter_list must contain positive iteration counts",
)
@icontract.require(lambda this_iter: _iteration_source(this_iter), "this_iter must start with a positive iteration count")
@icontract.ensure(
    lambda result, n_iter_list, this_iter: isinstance(result, list)
    and result == [int(value) for value in n_iter_list] + [int(np.asarray(this_iter)[0])],
    "iteration list must append this_iter[0]",
)
def cd_estimator_n_iter_with_target(
    n_iter_list: Sequence[int], this_iter: object
) -> list[int]:
    """Return the iteration-count list after appending one target count."""
    return [int(value) for value in n_iter_list] + [int(np.asarray(this_iter)[0])]
