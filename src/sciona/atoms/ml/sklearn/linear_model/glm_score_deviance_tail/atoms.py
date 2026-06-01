"""Sklearn GLM score deviance-tail atoms."""

from __future__ import annotations

from numbers import Real

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_glm_score_constant_average,
    witness_glm_score_d2_from_deviances,
    witness_glm_score_null_raw_prediction,
    witness_glm_score_sample_weight_check_args,
    witness_glm_score_sample_weight_check_kwargs,
    witness_glm_score_target_range_error_message,
    witness_glm_score_y_check_array_kwargs,
)

_Y_CHECK_KWARGS_KEYS = {"dtype", "order", "ensure_2d"}


def _has_dtype(value: object) -> bool:
    return hasattr(value, "dtype")


def _not_none(value: object) -> bool:
    return value is not None


def _finite_vector(value: object) -> bool:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 1 and array.size >= 1 and np.all(np.isfinite(array)))


def _sample_weight_valid(sample_weight: object, n_samples: int) -> bool:
    if sample_weight is None:
        return True
    try:
        weights = np.asarray(sample_weight, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(weights.ndim == 1 and weights.shape[0] == int(n_samples) and np.all(np.isfinite(weights)) and np.all(weights >= 0.0) and np.sum(weights) > 0.0)


def _finite_scalar(value: object) -> bool:
    return bool(isinstance(value, Real) and not isinstance(value, bool) and np.isfinite(float(value)))


def _nonempty_string(value: object) -> bool:
    return bool(isinstance(value, str) and value)


def _d2_denominator_valid(deviance_null: float, constant: float) -> bool:
    return bool(_finite_scalar(deviance_null) and _finite_scalar(constant) and np.isfinite(float(deviance_null) + float(constant)) and (float(deviance_null) + float(constant)) != 0.0)


def _y_check_kwargs_valid(result: dict[str, object], raw_prediction: object) -> bool:
    return bool(
        set(result) == _Y_CHECK_KWARGS_KEYS
        and result["dtype"] == raw_prediction.dtype
        and result["order"] == "C"
        and result["ensure_2d"] is False
    )


def _null_prediction_valid(result: object, y: object) -> bool:
    values = np.asarray(result, dtype=np.float64)
    target = np.asarray(y)
    return bool(values.ndim == 1 and values.shape == (target.shape[0],) and np.all(np.isfinite(values)))


@register_atom(witness_glm_score_y_check_array_kwargs)
@icontract.require(lambda raw_prediction: _has_dtype(raw_prediction), "raw_prediction must expose dtype")
@icontract.ensure(
    lambda result, raw_prediction: _y_check_kwargs_valid(result, raw_prediction),
    "target check_array kwargs must match GLM score source",
)
def glm_score_y_check_array_kwargs(raw_prediction: object) -> dict[str, object]:
    """Return kwargs passed to check_array for GLM score targets."""
    return {"dtype": raw_prediction.dtype, "order": "C", "ensure_2d": False}


@register_atom(witness_glm_score_sample_weight_check_args)
@icontract.require(lambda sample_weight: _not_none(sample_weight), "sample_weight must be supplied")
@icontract.require(lambda X: _not_none(X), "X must be supplied")
@icontract.ensure(
    lambda result, sample_weight, X: isinstance(result, tuple) and len(result) == 2 and result[0] is sample_weight and result[1] is X,
    "sample-weight validation args must preserve identities",
)
def glm_score_sample_weight_check_args(sample_weight: object, X: object) -> tuple[object, object]:
    """Return positional args passed to _check_sample_weight during GLM score."""
    return (sample_weight, X)


@register_atom(witness_glm_score_sample_weight_check_kwargs)
@icontract.require(lambda y: _has_dtype(y), "y must expose dtype")
@icontract.ensure(lambda result, y: result == {"dtype": y.dtype}, "sample-weight validation dtype must come from y")
def glm_score_sample_weight_check_kwargs(y: object) -> dict[str, object]:
    """Return kwargs passed to _check_sample_weight during GLM score."""
    return {"dtype": y.dtype}


@register_atom(witness_glm_score_target_range_error_message)
@icontract.require(lambda loss_name: _nonempty_string(loss_name), "loss_name must be a nonempty string")
@icontract.ensure(
    lambda result, loss_name: result == f"Some value(s) of y are out of the valid range of the loss {loss_name}.",
    "target-range error message must match GLM score source",
)
def glm_score_target_range_error_message(loss_name: str) -> str:
    """Return the GLM score invalid-target-range error message."""
    return f"Some value(s) of y are out of the valid range of the loss {loss_name}."


@register_atom(witness_glm_score_constant_average)
@icontract.require(lambda constant_values: _finite_vector(constant_values), "constant values must be a finite vector")
@icontract.require(
    lambda constant_values, sample_weight: _sample_weight_valid(sample_weight, np.asarray(constant_values).shape[0]),
    "sample_weight must align with constant values",
)
@icontract.ensure(lambda result: _finite_scalar(result), "constant average must be finite")
def glm_score_constant_average(constant_values: NDArray[np.float64], sample_weight: object = None) -> float:
    """Return the weighted average of supplied constant-to-zero values."""
    return float(np.average(np.asarray(constant_values, dtype=np.float64), weights=sample_weight))


@register_atom(witness_glm_score_null_raw_prediction)
@icontract.require(lambda y: _finite_vector(y), "y must be a finite vector")
@icontract.require(lambda linked_mean: _finite_scalar(linked_mean), "linked_mean must be finite")
@icontract.ensure(lambda result, y: _null_prediction_valid(result, y), "null raw prediction must align with y")
def glm_score_null_raw_prediction(y: object, linked_mean: float) -> NDArray[np.float64]:
    """Return the null-model raw prediction vector from a supplied linked mean."""
    return np.asarray(np.tile(float(linked_mean), np.asarray(y).shape[0]), dtype=np.float64)


@register_atom(witness_glm_score_d2_from_deviances)
@icontract.require(lambda deviance: _finite_scalar(deviance), "deviance must be finite")
@icontract.require(lambda deviance_null, constant: _d2_denominator_valid(deviance_null, constant), "null deviance plus constant must be finite and nonzero")
@icontract.ensure(lambda result: _finite_scalar(result), "D2 score must be finite")
def glm_score_d2_from_deviances(deviance: float, deviance_null: float, constant: float) -> float:
    """Return the GLM D2 score from supplied deviances and constant."""
    return 1.0 - (float(deviance) + float(constant)) / (float(deviance_null) + float(constant))
