"""Sklearn LogisticRegressionCV final array packaging atoms."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_logistic_cv_C_array,
    witness_logistic_cv_l1_ratio_array,
    witness_logistic_cv_public_l1_ratios_array,
)


def _finite_numeric_array(value: object) -> bool:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        return False
    return bool(array.size >= 1 and np.issubdtype(array.dtype, np.number) and np.all(np.isfinite(array)))


def _nonempty_array(value: object) -> bool:
    try:
        array = np.asarray(value, dtype=object)
    except (TypeError, ValueError):
        return False
    return bool(array.size >= 1)


def _array_result_valid(result: NDArray[object], source: object) -> bool:
    expected = np.asarray(source)
    return bool(isinstance(result, np.ndarray) and result.dtype == expected.dtype and np.array_equal(result, expected))


@register_atom(witness_logistic_cv_C_array)
@icontract.require(lambda C_values: _finite_numeric_array(C_values), "C_values must be finite numeric values")
@icontract.ensure(lambda result, C_values: _array_result_valid(result, C_values), "C_ array must match np.asarray(C_values)")
def logistic_cv_C_array(C_values: object) -> NDArray[np.floating]:
    """Return final LogisticRegressionCV selected C_ values as an ndarray."""
    return np.asarray(C_values)


@register_atom(witness_logistic_cv_l1_ratio_array)
@icontract.require(lambda l1_ratio_values: _nonempty_array(l1_ratio_values), "l1_ratio_values must be nonempty")
@icontract.ensure(lambda result, l1_ratio_values: _array_result_valid(result, l1_ratio_values), "l1_ratio_ array must match np.asarray(l1_ratio_values)")
def logistic_cv_l1_ratio_array(l1_ratio_values: object) -> NDArray[object]:
    """Return final LogisticRegressionCV selected l1_ratio_ values as an ndarray."""
    return np.asarray(l1_ratio_values)


@register_atom(witness_logistic_cv_public_l1_ratios_array)
@icontract.require(lambda l1_ratios: _nonempty_array(l1_ratios), "l1_ratios must be nonempty")
@icontract.ensure(lambda result, l1_ratios: _array_result_valid(result, l1_ratios), "l1_ratios_ array must match np.asarray(l1_ratios)")
def logistic_cv_public_l1_ratios_array(l1_ratios: object) -> NDArray[object]:
    """Return final LogisticRegressionCV public l1_ratios_ grid as an ndarray."""
    return np.asarray(l1_ratios)
