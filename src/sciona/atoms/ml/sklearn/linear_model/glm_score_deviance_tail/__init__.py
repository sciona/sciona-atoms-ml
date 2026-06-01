"""Sklearn GLM score deviance-tail atoms."""

from __future__ import annotations

from .atoms import (
    glm_score_constant_average,
    glm_score_d2_from_deviances,
    glm_score_null_raw_prediction,
    glm_score_sample_weight_check_args,
    glm_score_sample_weight_check_kwargs,
    glm_score_target_range_error_message,
    glm_score_y_check_array_kwargs,
)

__all__ = [
    "glm_score_constant_average",
    "glm_score_d2_from_deviances",
    "glm_score_null_raw_prediction",
    "glm_score_sample_weight_check_args",
    "glm_score_sample_weight_check_kwargs",
    "glm_score_target_range_error_message",
    "glm_score_y_check_array_kwargs",
]
