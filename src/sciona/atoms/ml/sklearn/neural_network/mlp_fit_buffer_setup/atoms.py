"""MLP fit buffer-setup helper atoms adapted from scikit-learn."""

from __future__ import annotations

from collections.abc import Sequence

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_mlp_fit_coef_gradient_buffers,
    witness_mlp_fit_intercept_gradient_buffers,
    witness_mlp_fit_layer_units,
    witness_mlp_fit_targets_2d,
)


def _finite_targets(y: object) -> bool:
    try:
        values = np.asarray(y, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(values.ndim in {1, 2} and values.shape[0] >= 1 and np.all(np.isfinite(values)))


def _targets_2d_valid(result: NDArray[np.float64], y: object) -> bool:
    original = np.asarray(y, dtype=np.float64)
    values = np.asarray(result, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != original.shape[0] or not np.all(np.isfinite(values)):
        return False
    if original.ndim == 1:
        return bool(values.shape[1] == 1 and np.array_equal(values[:, 0], original))
    return bool(values.shape == original.shape and np.array_equal(values, original))


def _positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


def _hidden_layer_sizes_valid(values: object) -> bool:
    return bool(
        isinstance(values, Sequence)
        and not isinstance(values, (str, bytes))
        and all(_positive_int(value) for value in values)
    )


def _layer_units_valid(result: tuple[int, ...], n_features: int, hidden_layer_sizes: tuple[int, ...], n_outputs: int) -> bool:
    expected = (int(n_features), *tuple(int(value) for value in hidden_layer_sizes), int(n_outputs))
    return bool(isinstance(result, tuple) and result == expected and len(result) >= 2)


def _dtype_name_valid(dtype_name: object) -> bool:
    if not isinstance(dtype_name, str) or not dtype_name:
        return False
    try:
        np.dtype(dtype_name)
    except TypeError:
        return False
    return True


def _coef_gradient_buffers_valid(result: tuple[NDArray[np.generic], ...], layer_units: tuple[int, ...], dtype_name: str) -> bool:
    dtype = np.dtype(dtype_name)
    if not isinstance(result, tuple) or len(result) != len(layer_units) - 1:
        return False
    for buffer, fan_in, fan_out in zip(result, layer_units[:-1], layer_units[1:]):
        values = np.asarray(buffer)
        if values.shape != (int(fan_in), int(fan_out)) or values.dtype != dtype:
            return False
    return True


def _intercept_gradient_buffers_valid(result: tuple[NDArray[np.generic], ...], layer_units: tuple[int, ...], dtype_name: str) -> bool:
    dtype = np.dtype(dtype_name)
    if not isinstance(result, tuple) or len(result) != len(layer_units) - 1:
        return False
    for buffer, fan_out in zip(result, layer_units[1:]):
        values = np.asarray(buffer)
        if values.shape != (int(fan_out),) or values.dtype != dtype:
            return False
    return True


@register_atom(witness_mlp_fit_targets_2d)
@icontract.require(lambda y: _finite_targets(y), "y must be a finite 1D or 2D target array")
@icontract.ensure(lambda result, y: _targets_2d_valid(result, y), "result must match sklearn's 2D target reshape logic")
def mlp_fit_targets_2d(
    y: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Ensure sklearn's MLP fit loop sees a 2D target matrix."""
    values = np.asarray(y, dtype=np.float64)
    if values.ndim == 1:
        return np.asarray(values.reshape((-1, 1)), dtype=np.float64)
    return np.asarray(values, dtype=np.float64)


@register_atom(witness_mlp_fit_layer_units)
@icontract.require(lambda n_features: _positive_int(n_features), "n_features must be a positive integer")
@icontract.require(lambda hidden_layer_sizes: _hidden_layer_sizes_valid(hidden_layer_sizes), "hidden_layer_sizes must be a sequence of positive integers")
@icontract.require(lambda n_outputs: _positive_int(n_outputs), "n_outputs must be a positive integer")
@icontract.ensure(lambda result, n_features, hidden_layer_sizes, n_outputs: _layer_units_valid(result, n_features, tuple(int(v) for v in hidden_layer_sizes), n_outputs), "layer_units must prepend inputs and append outputs around the hidden widths")
def mlp_fit_layer_units(
    n_features: int,
    hidden_layer_sizes: tuple[int, ...],
    n_outputs: int,
) -> tuple[int, ...]:
    """Construct sklearn's layer-width sequence before MLP solver execution."""
    return (int(n_features), *tuple(int(value) for value in hidden_layer_sizes), int(n_outputs))


@register_atom(witness_mlp_fit_coef_gradient_buffers)
@icontract.require(lambda layer_units: _hidden_layer_sizes_valid(layer_units) and len(layer_units) >= 2, "layer_units must contain at least input and output widths")
@icontract.require(lambda dtype_name: _dtype_name_valid(dtype_name), "dtype_name must name a valid NumPy dtype")
@icontract.ensure(lambda result, layer_units, dtype_name: _coef_gradient_buffers_valid(result, tuple(int(v) for v in layer_units), dtype_name), "coefficient gradient buffers must match sklearn's per-layer shapes and dtype")
def mlp_fit_coef_gradient_buffers(
    layer_units: tuple[int, ...],
    dtype_name: str,
) -> tuple[NDArray[np.generic], ...]:
    """Allocate sklearn's coefficient-gradient buffers before MLP solver execution."""
    dtype = np.dtype(dtype_name)
    return tuple(
        np.empty((int(n_fan_in), int(n_fan_out)), dtype=dtype)
        for n_fan_in, n_fan_out in zip(layer_units[:-1], layer_units[1:])
    )


@register_atom(witness_mlp_fit_intercept_gradient_buffers)
@icontract.require(lambda layer_units: _hidden_layer_sizes_valid(layer_units) and len(layer_units) >= 2, "layer_units must contain at least input and output widths")
@icontract.require(lambda dtype_name: _dtype_name_valid(dtype_name), "dtype_name must name a valid NumPy dtype")
@icontract.ensure(lambda result, layer_units, dtype_name: _intercept_gradient_buffers_valid(result, tuple(int(v) for v in layer_units), dtype_name), "intercept gradient buffers must match sklearn's per-layer widths and dtype")
def mlp_fit_intercept_gradient_buffers(
    layer_units: tuple[int, ...],
    dtype_name: str,
) -> tuple[NDArray[np.generic], ...]:
    """Allocate sklearn's intercept-gradient buffers before MLP solver execution."""
    dtype = np.dtype(dtype_name)
    return tuple(
        np.empty(int(n_fan_out), dtype=dtype)
        for n_fan_out in layer_units[1:]
    )
