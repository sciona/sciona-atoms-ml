"""Multioutput fit bookkeeping helpers adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_multioutput_fit_output_count,
    witness_multioutput_fit_require_2d_targets,
    witness_multioutput_fit_require_base_fit_method,
    witness_multioutput_fit_require_sample_weight_support,
    witness_multioutput_fit_target_column,
)


def _flag_valid(value: object) -> bool:
    return isinstance(value, bool)


def _target_matrix_like(y: object) -> bool:
    try:
        values = np.asarray(y, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(values.ndim >= 1 and values.shape[0] >= 1 and np.all(np.isfinite(values)))


def _target_matrix_valid(y: object) -> bool:
    try:
        values = np.asarray(y, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(values.ndim == 2 and values.shape[0] >= 1 and values.shape[1] >= 1 and np.all(np.isfinite(values)))


def _positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


def _target_column_valid(result: object, y: object) -> bool:
    try:
        column = np.asarray(result, dtype=np.float64)
        values = np.asarray(y, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(column.ndim == 1 and values.ndim == 2 and column.shape == (values.shape[0],) and np.all(np.isfinite(column)))


@register_atom(witness_multioutput_fit_require_base_fit_method)
@icontract.require(lambda estimator_has_fit: _flag_valid(estimator_has_fit), "estimator_has_fit must be boolean")
@icontract.ensure(lambda result: _flag_valid(result), "result must be boolean")
def multioutput_fit_require_base_fit_method(estimator_has_fit: bool) -> bool:
    """Enforce sklearn's base-estimator fit-method requirement before multioutput fit."""
    if not estimator_has_fit:
        raise ValueError("The base estimator should implement a fit method")
    return estimator_has_fit


@register_atom(witness_multioutput_fit_require_2d_targets)
@icontract.require(lambda y: _target_matrix_like(y), "y must be a finite array-like target structure with at least one sample")
@icontract.ensure(lambda result: _target_matrix_valid(result), "validated fit targets must be a finite nonempty 2D matrix")
def multioutput_fit_require_2d_targets(y: NDArray[np.float64]) -> NDArray[np.float64]:
    """Require sklearn's 2D target shape for multioutput fit."""
    values = np.asarray(y, dtype=np.float64)
    if values.ndim == 1:
        raise ValueError(
            "y must have at least two dimensions for multi-output regression but has only one."
        )
    return np.asarray(values, dtype=np.float64)


@register_atom(witness_multioutput_fit_output_count)
@icontract.require(lambda y: _target_matrix_valid(y), "y must be a finite nonempty 2D target matrix")
@icontract.ensure(lambda result, y: _positive_int(result) and result == np.asarray(y).shape[1], "output count must equal the number of target columns")
def multioutput_fit_output_count(y: NDArray[np.float64]) -> int:
    """Return the number of output columns used by sklearn's multioutput fit loop."""
    return int(np.asarray(y, dtype=np.float64).shape[1])


@register_atom(witness_multioutput_fit_target_column)
@icontract.require(lambda y: _target_matrix_valid(y), "y must be a finite nonempty 2D target matrix")
@icontract.require(lambda y, output_idx: isinstance(output_idx, int) and not isinstance(output_idx, bool) and 0 <= output_idx < np.asarray(y).shape[1], "output_idx must select an existing output column")
@icontract.ensure(lambda result, y: _target_column_valid(result, y), "target column must be a finite 1D vector with one value per sample")
def multioutput_fit_target_column(
    y: NDArray[np.float64],
    output_idx: int,
) -> NDArray[np.float64]:
    """Select one output column for a single base-estimator fit call."""
    values = np.asarray(y, dtype=np.float64)
    return np.asarray(values[:, output_idx], dtype=np.float64)


@register_atom(witness_multioutput_fit_require_sample_weight_support)
@icontract.require(lambda sample_weight_provided: _flag_valid(sample_weight_provided), "sample_weight_provided must be boolean")
@icontract.require(lambda estimator_supports_sample_weight: _flag_valid(estimator_supports_sample_weight), "estimator_supports_sample_weight must be boolean")
@icontract.ensure(lambda result: _flag_valid(result), "result must be boolean")
def multioutput_fit_require_sample_weight_support(
    *,
    sample_weight_provided: bool,
    estimator_supports_sample_weight: bool,
) -> bool:
    """Enforce sklearn's sample-weight support guard for multioutput fit."""
    if sample_weight_provided and not estimator_supports_sample_weight:
        raise ValueError("Underlying estimator does not support sample weights.")
    return True
