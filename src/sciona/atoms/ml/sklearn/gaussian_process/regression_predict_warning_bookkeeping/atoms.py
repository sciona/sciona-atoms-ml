"""Gaussian-process regression predict warning helpers adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_gp_predict_negative_variance_mask,
    witness_gp_predict_negative_variance_warning_required,
    witness_gp_predict_nonnegative_variance,
)


def _finite_vector(values: object) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 1 and array.shape[0] >= 1 and np.all(np.isfinite(array)))


def _mask_like(result: object, source: object) -> bool:
    values = np.asarray(result)
    source_values = np.asarray(source, dtype=np.float64)
    return bool(values.shape == source_values.shape and values.dtype == np.bool_)


def _boolean(value: object) -> bool:
    return isinstance(value, bool)


def _mask_valid(mask: object, source: object) -> bool:
    values = np.asarray(mask)
    source_values = np.asarray(source, dtype=np.float64)
    return bool(values.shape == source_values.shape and values.dtype == np.bool_)


def _nonnegative_vector_like(result: object, source: object) -> bool:
    values = np.asarray(result, dtype=np.float64)
    source_values = np.asarray(source, dtype=np.float64)
    return bool(values.shape == source_values.shape and np.all(np.isfinite(values)) and np.all(values >= 0.0))


@register_atom(witness_gp_predict_negative_variance_mask)
@icontract.require(lambda y_var: _finite_vector(y_var), "y_var must be a finite nonempty vector")
@icontract.ensure(lambda result, y_var: _mask_like(result, y_var), "result must be a boolean mask aligned with y_var")
def gp_predict_negative_variance_mask(
    y_var: NDArray[np.float64],
) -> NDArray[np.bool_]:
    """Mark predictive variances that are negative before clipping."""
    return np.asarray(np.asarray(y_var, dtype=np.float64) < 0.0, dtype=np.bool_)


@register_atom(witness_gp_predict_negative_variance_warning_required)
@icontract.require(lambda negative_mask: isinstance(np.asarray(negative_mask), np.ndarray) and np.asarray(negative_mask).dtype == np.bool_ and np.asarray(negative_mask).ndim == 1 and np.asarray(negative_mask).shape[0] >= 1, "negative_mask must be a nonempty one-dimensional boolean vector")
@icontract.ensure(lambda result: _boolean(result), "result must be boolean")
def gp_predict_negative_variance_warning_required(
    negative_mask: NDArray[np.bool_],
) -> bool:
    """Decide whether GaussianProcessRegressor.predict would warn about negative variances."""
    return bool(np.any(np.asarray(negative_mask, dtype=np.bool_)))


@register_atom(witness_gp_predict_nonnegative_variance)
@icontract.require(lambda y_var: _finite_vector(y_var), "y_var must be a finite nonempty vector")
@icontract.require(lambda y_var, negative_mask: _mask_valid(negative_mask, y_var), "negative_mask must be a boolean vector aligned with y_var")
@icontract.ensure(lambda result, y_var: _nonnegative_vector_like(result, y_var), "result must be a finite nonnegative vector aligned with y_var")
def gp_predict_nonnegative_variance(
    y_var: NDArray[np.float64],
    negative_mask: NDArray[np.bool_],
) -> NDArray[np.float64]:
    """Clip predictive variances at zero using the supplied negative-variance mask."""
    variance = np.asarray(y_var, dtype=np.float64).copy()
    variance[np.asarray(negative_mask, dtype=np.bool_)] = 0.0
    return variance
