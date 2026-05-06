"""Sklearn coordinate-descent path-residual mono-output normalization atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_cd_path_residuals_mono_output_coefs,
    witness_cd_path_residuals_mono_output_y_offset,
    witness_cd_path_residuals_mono_output_y_test,
    witness_cd_path_residuals_use_mono_output_normalization,
)


def _positive_int(value: object) -> bool:
    return isinstance(value, (int, np.integer)) and int(value) >= 1


def _finite_numeric_array(value: object, ndim: int | None = None) -> bool:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        return False
    return bool(
        np.issubdtype(array.dtype, np.number)
        and np.all(np.isfinite(array))
        and (ndim is None or array.ndim == ndim)
    )


def _mono_output_coefs_valid(result: object, coefs: object) -> bool:
    if not (_finite_numeric_array(result, ndim=3) and _finite_numeric_array(coefs, ndim=2)):
        return False
    result_values = np.asarray(result)
    coef_values = np.asarray(coefs)
    return bool(result_values.shape == (1,) + coef_values.shape and np.array_equal(result_values[0], coef_values))


def _mono_output_y_offset_valid(result: object, y_offset: object) -> bool:
    if not (_finite_numeric_array(result, ndim=1) and _finite_numeric_array(y_offset)):
        return False
    result_values = np.asarray(result)
    expected = np.atleast_1d(np.asarray(y_offset))
    return bool(np.array_equal(result_values, expected))


def _mono_output_y_test_valid(result: object, y_test: object) -> bool:
    if not (_finite_numeric_array(result, ndim=2) and _finite_numeric_array(y_test, ndim=1)):
        return False
    result_values = np.asarray(result)
    y_test_values = np.asarray(y_test)
    return bool(result_values.shape == (y_test_values.shape[0], 1) and np.array_equal(result_values[:, 0], y_test_values))


@register_atom(witness_cd_path_residuals_use_mono_output_normalization)
@icontract.require(lambda y_ndim: _positive_int(y_ndim), "y_ndim must be positive")
@icontract.ensure(
    lambda result, y_ndim: isinstance(result, bool) and result == (int(y_ndim) == 1),
    "mono-output normalization branch must match y.ndim == 1",
)
def cd_path_residuals_use_mono_output_normalization(y_ndim: int) -> bool:
    """Return whether _path_residuals applies mono-output normalization after the path callback."""
    return int(y_ndim) == 1


@register_atom(witness_cd_path_residuals_mono_output_coefs)
@icontract.require(lambda coefs: _finite_numeric_array(coefs, ndim=2), "coefs must be a finite numeric rank-2 array")
@icontract.ensure(
    lambda result, coefs: _mono_output_coefs_valid(result, coefs),
    "mono-output coefficient normalization must prepend one output axis",
)
def cd_path_residuals_mono_output_coefs(coefs: NDArray[np.floating]) -> NDArray[np.floating]:
    """Return sklearn's coefs[np.newaxis, :, :] mono-output normalization."""
    return coefs[np.newaxis, :, :]


@register_atom(witness_cd_path_residuals_mono_output_y_offset)
@icontract.require(lambda y_offset: _finite_numeric_array(y_offset), "y_offset must be finite and numeric")
@icontract.ensure(
    lambda result, y_offset: _mono_output_y_offset_valid(result, y_offset),
    "mono-output y_offset normalization must match np.atleast_1d",
)
def cd_path_residuals_mono_output_y_offset(y_offset: object) -> NDArray[np.floating]:
    """Return sklearn's np.atleast_1d(y_offset) mono-output normalization."""
    return np.atleast_1d(np.asarray(y_offset))


@register_atom(witness_cd_path_residuals_mono_output_y_test)
@icontract.require(lambda y_test: _finite_numeric_array(y_test, ndim=1), "y_test must be a finite numeric rank-1 array")
@icontract.ensure(
    lambda result, y_test: _mono_output_y_test_valid(result, y_test),
    "mono-output y_test normalization must append one column axis",
)
def cd_path_residuals_mono_output_y_test(y_test: NDArray[np.floating]) -> NDArray[np.floating]:
    """Return sklearn's y_test[:, np.newaxis] mono-output normalization."""
    return y_test[:, np.newaxis]
