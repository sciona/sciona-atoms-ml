"""Multioutput partial-fit bookkeeping helpers adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_multioutput_partial_fit_class_vector,
    witness_multioutput_partial_fit_first_call,
    witness_multioutput_partial_fit_use_base_estimator,
)

ClassVector = NDArray[np.float64]
ClassVectorTuple = tuple[ClassVector, ...]


def _flag_valid(value: object) -> bool:
    return isinstance(value, bool)


def _class_vector_valid(values: object) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(
        array.ndim == 1
        and array.shape[0] >= 1
        and np.all(np.isfinite(array))
        and np.unique(array).shape[0] == array.shape[0]
    )


def _class_vector_tuple_or_none(values: object) -> bool:
    return values is None or (
        isinstance(values, tuple)
        and len(values) >= 1
        and all(_class_vector_valid(value) for value in values)
    )


def _class_vector_result_valid(result: object, classes_by_output: object, output_idx: int) -> bool:
    if classes_by_output is None:
        return result is None
    try:
        observed = np.asarray(result, dtype=np.float64)
        expected = np.asarray(classes_by_output[output_idx], dtype=np.float64)
    except (TypeError, ValueError, IndexError):
        return False
    return bool(observed.ndim == 1 and np.array_equal(observed, expected))


@register_atom(witness_multioutput_partial_fit_first_call)
@icontract.require(lambda has_estimators: _flag_valid(has_estimators), "has_estimators must be boolean")
@icontract.ensure(lambda result: _flag_valid(result), "first-call flag must be boolean")
def multioutput_partial_fit_first_call(has_estimators: bool) -> bool:
    """Return whether multioutput partial_fit is entering before estimators_ exists."""
    return not has_estimators


@register_atom(witness_multioutput_partial_fit_use_base_estimator)
@icontract.require(lambda first_time: _flag_valid(first_time), "first_time must be boolean")
@icontract.ensure(lambda result: _flag_valid(result), "base-estimator selection flag must be boolean")
def multioutput_partial_fit_use_base_estimator(first_time: bool) -> bool:
    """Return whether one multioutput partial_fit worker should receive the base estimator template."""
    return bool(first_time)


@register_atom(witness_multioutput_partial_fit_class_vector)
@icontract.require(lambda classes_by_output: _class_vector_tuple_or_none(classes_by_output), "classes_by_output must be None or a nonempty tuple of finite unique class vectors")
@icontract.require(
    lambda classes_by_output, output_idx: isinstance(output_idx, int)
    and not isinstance(output_idx, bool)
    and (classes_by_output is None or 0 <= output_idx < len(classes_by_output)),
    "output_idx must select an existing class vector when classes_by_output is provided",
)
@icontract.ensure(lambda result, classes_by_output, output_idx: _class_vector_result_valid(result, classes_by_output, output_idx), "result must preserve the selected classes vector or None")
def multioutput_partial_fit_class_vector(
    classes_by_output: ClassVectorTuple | None,
    output_idx: int,
) -> ClassVector | None:
    """Return the per-output classes vector sklearn routes into one partial_fit worker."""
    if classes_by_output is None:
        return None
    return np.asarray(classes_by_output[output_idx], dtype=np.float64)
