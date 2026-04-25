"""Gaussian-process kernel hyperparameter bookkeeping atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_gp_kernel_bound_warning_records,
    witness_gp_kernel_bounds,
    witness_gp_kernel_theta,
    witness_gp_kernel_values_from_theta,
)

ValueBlock = NDArray[np.float64]
BoundsBlock = NDArray[np.float64] | str
WarningRecord = tuple[str, int, str, float]


def _positive_value_block(value_block: object) -> bool:
    try:
        values = np.atleast_1d(np.asarray(value_block, dtype=np.float64))
    except (TypeError, ValueError):
        return False
    return bool(values.ndim == 1 and values.size >= 1 and np.all(np.isfinite(values)) and np.all(values > 0.0))


def _value_blocks_valid(value_blocks: object) -> bool:
    return bool(isinstance(value_blocks, tuple) and len(value_blocks) >= 1 and all(_positive_value_block(block) for block in value_blocks))


def _fixed_tuple_valid(fixed: object, n_blocks: int) -> bool:
    return bool(
        isinstance(fixed, tuple)
        and len(fixed) == n_blocks
        and all(isinstance(item, bool) for item in fixed)
    )


def _names_valid(names: object, n_blocks: int) -> bool:
    return bool(
        isinstance(names, tuple)
        and len(names) == n_blocks
        and all(isinstance(item, str) and item != "" for item in names)
    )


def _numeric_bounds_block_valid(bounds_block: object) -> bool:
    try:
        values = np.asarray(bounds_block, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(
        values.ndim == 2
        and values.shape[1] == 2
        and np.all(np.isfinite(values))
        and np.all(values > 0.0)
        and np.all(values[:, 0] <= values[:, 1])
    )


def _bounds_blocks_valid(bounds_blocks: object, fixed: tuple[bool, ...]) -> bool:
    return bool(
        isinstance(bounds_blocks, tuple)
        and len(bounds_blocks) == len(fixed)
        and all(
            ((block == "fixed") if is_fixed else _numeric_bounds_block_valid(block))
            for block, is_fixed in zip(bounds_blocks, fixed)
        )
    )


def _theta_valid(theta: object) -> bool:
    try:
        values = np.asarray(theta, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(values.ndim == 1 and np.all(np.isfinite(values)))


def _nonfixed_dim_count(value_blocks: tuple[ValueBlock, ...], fixed: tuple[bool, ...]) -> int:
    return int(
        sum(
            np.atleast_1d(np.asarray(block, dtype=np.float64)).shape[0]
            for block, is_fixed in zip(value_blocks, fixed)
            if not is_fixed
        )
    )


def _theta_matches_blocks(theta: NDArray[np.float64], value_blocks: tuple[ValueBlock, ...], fixed: tuple[bool, ...]) -> bool:
    return bool(np.asarray(theta, dtype=np.float64).shape == (_nonfixed_dim_count(value_blocks, fixed),))


def _theta_result_valid(result: NDArray[np.float64], value_blocks: tuple[ValueBlock, ...], fixed: tuple[bool, ...]) -> bool:
    values = np.asarray(result, dtype=np.float64)
    return bool(values.shape == (_nonfixed_dim_count(value_blocks, fixed),) and np.all(np.isfinite(values)))


def _values_result_valid(result: tuple[ValueBlock, ...], value_blocks: tuple[ValueBlock, ...]) -> bool:
    if not isinstance(result, tuple) or len(result) != len(value_blocks):
        return False
    for actual, original in zip(result, value_blocks):
        actual_values = np.asarray(actual, dtype=np.float64)
        original_values = np.atleast_1d(np.asarray(original, dtype=np.float64))
        if actual_values.shape != original_values.shape or not np.all(np.isfinite(actual_values)) or not np.all(actual_values > 0.0):
            return False
    return True


def _bounds_result_valid(result: NDArray[np.float64], value_blocks: tuple[ValueBlock, ...], fixed: tuple[bool, ...]) -> bool:
    values = np.asarray(result, dtype=np.float64)
    n_dims = _nonfixed_dim_count(value_blocks, fixed)
    if n_dims == 0:
        return bool(values.ndim == 1 and values.shape == (0,))
    return bool(values.shape == (n_dims, 2) and np.all(np.isfinite(values)))


def _warning_records_valid(result: tuple[WarningRecord, ...]) -> bool:
    return bool(
        isinstance(result, tuple)
        and all(
            isinstance(record, tuple)
            and len(record) == 4
            and isinstance(record[0], str)
            and record[0] != ""
            and isinstance(record[1], int)
            and not isinstance(record[1], bool)
            and record[1] >= 0
            and record[2] in {"lower", "upper"}
            and isinstance(record[3], float)
            and np.isfinite(record[3])
            and record[3] > 0.0
            for record in result
        )
    )


@register_atom(witness_gp_kernel_theta)
@icontract.require(lambda value_blocks: _value_blocks_valid(value_blocks), "value_blocks must be a nonempty tuple of finite positive 1D hyperparameter arrays")
@icontract.require(lambda value_blocks, fixed: _fixed_tuple_valid(fixed, len(value_blocks)), "fixed must match the number of hyperparameter blocks")
@icontract.ensure(lambda result, value_blocks, fixed: _theta_result_valid(result, value_blocks, fixed), "theta must flatten exactly the non-fixed hyperparameter blocks")
def gp_kernel_theta(
    value_blocks: tuple[ValueBlock, ...],
    fixed: tuple[bool, ...],
) -> NDArray[np.float64]:
    """Flatten non-fixed kernel hyperparameters into sklearn's log-theta vector."""
    theta_blocks = [
        np.atleast_1d(np.asarray(block, dtype=np.float64))
        for block, is_fixed in zip(value_blocks, fixed)
        if not is_fixed
    ]
    if not theta_blocks:
        return np.array([], dtype=np.float64)
    return np.log(np.hstack(theta_blocks))


@register_atom(witness_gp_kernel_values_from_theta)
@icontract.require(lambda theta: _theta_valid(theta), "theta must be a finite one-dimensional vector")
@icontract.require(lambda value_blocks: _value_blocks_valid(value_blocks), "value_blocks must be a nonempty tuple of finite positive 1D hyperparameter arrays")
@icontract.require(lambda value_blocks, fixed: _fixed_tuple_valid(fixed, len(value_blocks)), "fixed must match the number of hyperparameter blocks")
@icontract.require(lambda theta, value_blocks, fixed: _theta_matches_blocks(theta, value_blocks, fixed), "theta length must match the total number of non-fixed hyperparameter elements")
@icontract.ensure(lambda result, value_blocks: _values_result_valid(result, value_blocks), "updated hyperparameter blocks must preserve the input block structure")
def gp_kernel_values_from_theta(
    theta: NDArray[np.float64],
    value_blocks: tuple[ValueBlock, ...],
    fixed: tuple[bool, ...],
) -> tuple[ValueBlock, ...]:
    """Reconstruct per-hyperparameter value blocks from sklearn's log-theta vector."""
    theta_values = np.asarray(theta, dtype=np.float64)
    updated: list[ValueBlock] = []
    offset = 0
    for block, is_fixed in zip(value_blocks, fixed):
        current = np.atleast_1d(np.asarray(block, dtype=np.float64))
        if is_fixed:
            updated.append(current.copy())
            continue
        width = current.shape[0]
        updated.append(np.exp(theta_values[offset : offset + width]).reshape(current.shape))
        offset += width

    if offset != len(theta_values):
        raise ValueError(
            "theta has not the correct number of entries."
            f" Should be {offset}; given are {len(theta_values)}"
        )
    return tuple(np.asarray(block, dtype=np.float64) for block in updated)


@register_atom(witness_gp_kernel_bounds)
@icontract.require(lambda bounds_blocks, fixed: _fixed_tuple_valid(fixed, len(bounds_blocks)) and _bounds_blocks_valid(bounds_blocks, fixed), "bounds_blocks must provide one positive finite (n_elements, 2) bounds matrix per non-fixed hyperparameter block")
@icontract.require(lambda bounds_blocks, fixed: _fixed_tuple_valid(fixed, len(bounds_blocks)), "fixed must match the number of bounds blocks")
@icontract.ensure(lambda result, bounds_blocks, fixed: _bounds_result_valid(result, tuple(np.ones(np.asarray(block).shape[0], dtype=np.float64) for block, is_fixed in zip(bounds_blocks, fixed) if not is_fixed), tuple(False for block, is_fixed in zip(bounds_blocks, fixed) if not is_fixed)), "bounds must flatten exactly the non-fixed hyperparameter bounds")
def gp_kernel_bounds(
    bounds_blocks: tuple[BoundsBlock, ...],
    fixed: tuple[bool, ...],
) -> NDArray[np.float64]:
    """Flatten non-fixed kernel hyperparameter bounds into sklearn's log-bounds matrix."""
    free_bounds = [
        np.asarray(block, dtype=np.float64)
        for block, is_fixed in zip(bounds_blocks, fixed)
        if not is_fixed
    ]
    if not free_bounds:
        return np.array([], dtype=np.float64)
    return np.log(np.vstack(free_bounds))


@register_atom(witness_gp_kernel_bound_warning_records)
@icontract.require(lambda theta: _theta_valid(theta), "theta must be a finite one-dimensional vector")
@icontract.require(lambda bounds_blocks, fixed: _fixed_tuple_valid(fixed, len(bounds_blocks)) and _bounds_blocks_valid(bounds_blocks, fixed), "bounds_blocks and fixed must have matching lengths and valid positive bounds")
@icontract.require(lambda bounds_blocks, names: _names_valid(names, len(bounds_blocks)), "names must match the number of hyperparameter blocks")
@icontract.require(lambda theta, bounds_blocks, fixed: np.asarray(theta, dtype=np.float64).shape == (sum(np.asarray(block).shape[0] for block, is_fixed in zip(bounds_blocks, fixed) if not is_fixed),), "theta length must match the total number of non-fixed hyperparameter elements")
@icontract.ensure(lambda result: _warning_records_valid(result), "warning records must be a tuple of (name, dim, side, bound_value) entries")
def gp_kernel_bound_warning_records(
    theta: NDArray[np.float64],
    bounds_blocks: tuple[BoundsBlock, ...],
    fixed: tuple[bool, ...],
    names: tuple[str, ...],
) -> tuple[WarningRecord, ...]:
    """Return the warning contexts sklearn emits when fitted kernel theta is close to its bounds."""
    theta_values = np.asarray(theta, dtype=np.float64)
    if theta_values.size == 0:
        return ()

    log_bounds = gp_kernel_bounds(bounds_blocks, fixed)
    list_close = np.isclose(log_bounds, np.atleast_2d(theta_values).T)

    records: list[WarningRecord] = []
    theta_index = 0
    for name, block, is_fixed in zip(names, bounds_blocks, fixed):
        if is_fixed:
            continue
        bounds_array = np.asarray(block, dtype=np.float64)
        for dim in range(bounds_array.shape[0]):
            if list_close[theta_index, 0]:
                records.append((name, dim, "lower", float(bounds_array[dim, 0])))
            elif list_close[theta_index, 1]:
                records.append((name, dim, "upper", float(bounds_array[dim, 1])))
            theta_index += 1
    return tuple(records)
