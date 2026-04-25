"""MLP LBFGS bookkeeping helper atoms adapted from scikit-learn."""

from __future__ import annotations

from collections.abc import Sequence

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_mlp_lbfgs_coef_indptr,
    witness_mlp_lbfgs_intercept_indptr,
    witness_mlp_lbfgs_iprint,
    witness_mlp_lbfgs_pack_parameters,
)

CoefSlice = tuple[int, int, tuple[int, int]]
InterceptSlice = tuple[int, int]


def _positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


def _layer_units_valid(layer_units: object) -> bool:
    return bool(
        isinstance(layer_units, Sequence)
        and not isinstance(layer_units, (str, bytes))
        and len(layer_units) >= 2
        and all(_positive_int(value) for value in layer_units)
    )


def _coef_indptr_valid(result: tuple[CoefSlice, ...], layer_units: tuple[int, ...]) -> bool:
    if not isinstance(result, tuple) or len(result) != len(layer_units) - 1:
        return False
    start = 0
    for index, entry in enumerate(result):
        if not (isinstance(entry, tuple) and len(entry) == 3):
            return False
        slice_start, slice_end, shape = entry
        expected_shape = (int(layer_units[index]), int(layer_units[index + 1]))
        if (
            not _positive_int(slice_end)
            or not isinstance(slice_start, int)
            or slice_start != start
            or not isinstance(shape, tuple)
            or shape != expected_shape
            or slice_end - slice_start != expected_shape[0] * expected_shape[1]
        ):
            return False
        start = slice_end
    return True


def _intercept_indptr_valid(
    result: tuple[InterceptSlice, ...],
    layer_units: tuple[int, ...],
    coef_indptr: tuple[CoefSlice, ...],
) -> bool:
    if not isinstance(result, tuple) or len(result) != len(layer_units) - 1:
        return False
    start = int(coef_indptr[-1][1])
    for index, entry in enumerate(result):
        if not (isinstance(entry, tuple) and len(entry) == 2):
            return False
        slice_start, slice_end = entry
        expected_width = int(layer_units[index + 1])
        if (
            not isinstance(slice_start, int)
            or not isinstance(slice_end, int)
            or slice_start != start
            or slice_end - slice_start != expected_width
        ):
            return False
        start = slice_end
    return True


def _finite_matrix(values: object) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 2 and array.size >= 1 and np.all(np.isfinite(array)))


def _finite_vector(values: object) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 1 and array.size >= 1 and np.all(np.isfinite(array)))


def _parameter_blocks_valid(coefs: object, intercepts: object) -> bool:
    return bool(
        isinstance(coefs, tuple)
        and isinstance(intercepts, tuple)
        and len(coefs) >= 1
        and len(coefs) == len(intercepts)
        and all(_finite_matrix(coef) for coef in coefs)
        and all(_finite_vector(intercept) for intercept in intercepts)
        and all(np.asarray(coef).shape[1] == np.asarray(intercept).shape[0] for coef, intercept in zip(coefs, intercepts))
    )


@register_atom(witness_mlp_lbfgs_coef_indptr)
@icontract.require(lambda layer_units: _layer_units_valid(layer_units), "layer_units must contain at least two positive layer sizes")
@icontract.ensure(lambda result, layer_units: _coef_indptr_valid(result, tuple(int(value) for value in layer_units)), "coefficient index slices must match sklearn's LBFGS parameter layout")
def mlp_lbfgs_coef_indptr(
    layer_units: tuple[int, ...],
) -> tuple[CoefSlice, ...]:
    """Build sklearn's coefficient slice table for MLP LBFGS packing."""
    start = 0
    coef_indptr: list[CoefSlice] = []
    for i in range(len(layer_units) - 1):
        n_fan_in, n_fan_out = int(layer_units[i]), int(layer_units[i + 1])
        end = start + (n_fan_in * n_fan_out)
        coef_indptr.append((start, end, (n_fan_in, n_fan_out)))
        start = end
    return tuple(coef_indptr)


@register_atom(witness_mlp_lbfgs_intercept_indptr)
@icontract.require(lambda layer_units: _layer_units_valid(layer_units), "layer_units must contain at least two positive layer sizes")
@icontract.require(lambda layer_units, coef_indptr: _coef_indptr_valid(coef_indptr, tuple(int(value) for value in layer_units)), "coef_indptr must match layer_units")
@icontract.ensure(lambda result, layer_units, coef_indptr: _intercept_indptr_valid(result, tuple(int(value) for value in layer_units), coef_indptr), "intercept slices must continue sklearn's LBFGS parameter layout")
def mlp_lbfgs_intercept_indptr(
    layer_units: tuple[int, ...],
    coef_indptr: tuple[CoefSlice, ...],
) -> tuple[InterceptSlice, ...]:
    """Build sklearn's intercept slice table for MLP LBFGS packing."""
    start = int(coef_indptr[-1][1])
    intercept_indptr: list[InterceptSlice] = []
    for i in range(len(layer_units) - 1):
        end = start + int(layer_units[i + 1])
        intercept_indptr.append((start, end))
        start = end
    return tuple(intercept_indptr)


@register_atom(witness_mlp_lbfgs_pack_parameters)
@icontract.require(lambda coefs, intercepts: _parameter_blocks_valid(coefs, intercepts), "coefs and intercepts must be aligned finite parameter blocks")
@icontract.ensure(lambda result: isinstance(result, np.ndarray) and result.ndim == 1 and result.size >= 1 and np.all(np.isfinite(result)), "packed parameters must be a finite 1D vector")
def mlp_lbfgs_pack_parameters(
    coefs: tuple[NDArray[np.float64], ...],
    intercepts: tuple[NDArray[np.float64], ...],
) -> NDArray[np.float64]:
    """Pack MLP coefficients and intercepts into sklearn's flat LBFGS parameter vector."""
    return np.hstack([np.asarray(block, dtype=np.float64).ravel() for block in coefs + intercepts])


@register_atom(witness_mlp_lbfgs_iprint)
@icontract.require(lambda verbose: isinstance(verbose, (bool, int)) and not isinstance(verbose, np.bool_), "verbose must be bool or int")
@icontract.ensure(lambda result: isinstance(result, int), "iprint must be an integer")
def mlp_lbfgs_iprint(
    verbose: bool | int,
) -> int:
    """Resolve sklearn's LBFGS iprint option from the MLP verbose flag."""
    if verbose is True or int(verbose) >= 1:
        return 1
    return -1
