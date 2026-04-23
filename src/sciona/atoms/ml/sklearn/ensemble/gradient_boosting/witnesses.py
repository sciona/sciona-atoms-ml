"""Ghost witnesses for sklearn gradient-boosting helper atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_gradient_boosting_safe_divide(
    numerator: float,
    denominator: float,
) -> float:
    """Describe a scalar safe-division result."""
    del numerator, denominator
    return 0.0


def witness_gradient_boosting_huber_delta(
    y_true: AbstractArray,
    raw_prediction: AbstractArray,
    sample_weight: AbstractArray,
    *,
    quantile: float = 0.9,
) -> float:
    """Describe the scalar Huber delta selected from weighted residual magnitudes."""
    del raw_prediction, sample_weight, quantile
    if len(y_true.shape) != 1:
        raise ValueError("y_true must be one-dimensional")
    return 0.0
