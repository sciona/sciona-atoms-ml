"""Regressor-side MLP helper atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import r2_score

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_mlp_regressor_predictions,
    witness_mlp_regressor_r2_score,
    witness_mlp_regressor_targets,
)

RegressorArray = NDArray[np.float64]


def _numeric_array_valid(values: object) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(
        array.ndim in {1, 2}
        and array.shape[0] >= 1
        and (array.ndim == 1 or array.shape[1] >= 1)
    )


def _finite_array_valid(values: object) -> bool:
    if not _numeric_array_valid(values):
        return False
    array = np.asarray(values, dtype=np.float64)
    return bool(np.all(np.isfinite(array)))


def _prediction_matrix_valid(values: object) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 2 and array.shape[0] >= 1 and array.shape[1] >= 1 and np.all(np.isfinite(array)))


def _sample_weight_valid(sample_weight: object, n_samples: int) -> bool:
    if sample_weight is None:
        return True
    try:
        values = np.asarray(sample_weight, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(values.ndim == 1 and values.shape[0] == n_samples and np.all(np.isfinite(values)))


def _same_shape(y_true: object, y_pred: object) -> bool:
    return np.asarray(y_true, dtype=np.float64).shape == np.asarray(y_pred, dtype=np.float64).shape


def _target_result_valid(result: RegressorArray, y: RegressorArray) -> bool:
    values = np.asarray(result, dtype=np.float64)
    source = np.asarray(y, dtype=np.float64)
    if source.ndim == 2 and source.shape[1] == 1:
        return bool(values.ndim == 1 and values.shape == (source.shape[0],) and np.array_equal(values, source[:, 0]))
    return bool(values.shape == source.shape and np.array_equal(values, source))


def _prediction_result_valid(result: RegressorArray, y_pred: RegressorArray) -> bool:
    values = np.asarray(result, dtype=np.float64)
    source = np.asarray(y_pred, dtype=np.float64)
    if source.shape[1] == 1:
        return bool(values.ndim == 1 and values.shape == (source.shape[0],) and np.array_equal(values, source[:, 0]))
    return bool(values.shape == source.shape and np.array_equal(values, source))


def _score_result_valid(result: float) -> bool:
    return bool(isinstance(result, float) and (np.isnan(result) or np.isfinite(result)))


@register_atom(witness_mlp_regressor_targets)
@icontract.require(lambda y: _finite_array_valid(y), "y must be a finite 1D or 2D numeric target array")
@icontract.ensure(lambda result, y: _target_result_valid(result, y), "target output must preserve values and flatten a single target column")
def mlp_regressor_targets(
    y: RegressorArray,
) -> RegressorArray:
    """Apply MLPRegressor's single-column target flattening from _validate_input."""
    values = np.asarray(y, dtype=np.float64)
    if values.ndim == 2 and values.shape[1] == 1:
        return np.asarray(values[:, 0], dtype=np.float64)
    return np.asarray(values, dtype=np.float64)


@register_atom(witness_mlp_regressor_predictions)
@icontract.require(lambda y_pred: _prediction_matrix_valid(y_pred), "y_pred must be a finite sample-by-output prediction matrix")
@icontract.ensure(lambda result, y_pred: _prediction_result_valid(result, y_pred), "prediction output must preserve values and flatten a single output column")
def mlp_regressor_predictions(
    y_pred: RegressorArray,
) -> RegressorArray:
    """Apply MLPRegressor's one-output prediction flattening."""
    values = np.asarray(y_pred, dtype=np.float64)
    if values.shape[1] == 1:
        return np.asarray(values[:, 0], dtype=np.float64)
    return np.asarray(values, dtype=np.float64)


@register_atom(witness_mlp_regressor_r2_score)
@icontract.require(lambda y_true: _finite_array_valid(y_true), "y_true must be a finite 1D or 2D numeric array")
@icontract.require(lambda y_pred: _numeric_array_valid(y_pred), "y_pred must be a numeric 1D or 2D array")
@icontract.require(lambda y_true, y_pred: _same_shape(y_true, y_pred), "y_true and y_pred must have matching shapes")
@icontract.require(lambda y_true, sample_weight=None: _sample_weight_valid(sample_weight, np.asarray(y_true, dtype=np.float64).shape[0]), "sample_weight must be None or a finite vector matching the sample count")
@icontract.ensure(lambda result: _score_result_valid(result), "score result must be a finite float or NaN")
def mlp_regressor_r2_score(
    y_true: RegressorArray,
    y_pred: RegressorArray,
    *,
    sample_weight: NDArray[np.float64] | None = None,
) -> float:
    """Compute MLPRegressor's score helper with the NaN/Inf prediction short-circuit."""
    true_values = np.asarray(y_true, dtype=np.float64)
    pred_values = np.asarray(y_pred, dtype=np.float64)
    if np.isnan(pred_values).any() or np.isinf(pred_values).any():
        return float("nan")
    return float(r2_score(true_values, pred_values, sample_weight=sample_weight))
