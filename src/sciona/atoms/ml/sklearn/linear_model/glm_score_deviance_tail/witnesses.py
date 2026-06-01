"""Ghost witnesses for sklearn GLM score deviance-tail atoms."""

from __future__ import annotations

import numpy as np


def witness_glm_score_y_check_array_kwargs(raw_prediction: object) -> dict[str, object]:
    """Describe check_array kwargs for GLM score targets."""
    return {"dtype": raw_prediction.dtype, "order": "C", "ensure_2d": False}


def witness_glm_score_sample_weight_check_args(sample_weight: object, X: object) -> tuple[object, object]:
    """Describe positional args for GLM score sample-weight validation."""
    return (sample_weight, X)


def witness_glm_score_sample_weight_check_kwargs(y: object) -> dict[str, object]:
    """Describe keyword args for GLM score sample-weight validation."""
    return {"dtype": y.dtype}


def witness_glm_score_target_range_error_message(loss_name: str) -> str:
    """Describe the GLM score invalid-target-range error message."""
    return f"Some value(s) of y are out of the valid range of the loss {loss_name}."


def witness_glm_score_constant_average(constant_values: object, sample_weight: object) -> float:
    """Describe weighted averaging of supplied constant-to-zero values."""
    return float(np.average(constant_values, weights=sample_weight))


def witness_glm_score_null_raw_prediction(y: object, linked_mean: float) -> object:
    """Describe null-model raw prediction tiling from a supplied linked mean."""
    return np.tile(float(linked_mean), np.asarray(y).shape[0])


def witness_glm_score_d2_from_deviances(deviance: float, deviance_null: float, constant: float) -> float:
    """Describe the final GLM D2 score from supplied deviances."""
    return 1.0 - (float(deviance) + float(constant)) / (float(deviance_null) + float(constant))
