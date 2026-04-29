"""Shape descriptions for sklearn MLP regressor input and output helpers."""

from __future__ import annotations

import numpy as np

from sciona.ghost.abstract import AbstractArray


def _check_array(values: AbstractArray, name: str) -> tuple[int, int | None]:
    if len(values.shape) not in {1, 2}:
        raise ValueError(f"{name} must be 1D or 2D")
    n_samples = int(values.shape[0])
    if n_samples < 1:
        raise ValueError(f"{name} must be nonempty")
    if len(values.shape) == 1:
        return n_samples, None
    n_outputs = int(values.shape[1])
    if n_outputs < 1:
        raise ValueError(f"{name} must have at least one output column")
    return n_samples, n_outputs


def witness_mlp_regressor_targets(
    y: AbstractArray,
) -> AbstractArray:
    """Describe MLPRegressor target reshaping from _validate_input."""
    n_samples, n_outputs = _check_array(y, "y")
    if n_outputs == 1:
        return AbstractArray(shape=(n_samples,), dtype=np.float64)
    if n_outputs is None:
        return AbstractArray(shape=(n_samples,), dtype=np.float64)
    return AbstractArray(shape=(n_samples, n_outputs), dtype=np.float64)


def witness_mlp_regressor_predictions(
    y_pred: AbstractArray,
) -> AbstractArray:
    """Describe MLPRegressor prediction output shaping."""
    if len(y_pred.shape) != 2:
        raise ValueError("y_pred must be 2D")
    n_samples = int(y_pred.shape[0])
    n_outputs = int(y_pred.shape[1])
    if n_samples < 1 or n_outputs < 1:
        raise ValueError("y_pred must be nonempty")
    if n_outputs == 1:
        return AbstractArray(shape=(n_samples,), dtype=np.float64)
    return AbstractArray(shape=(n_samples, n_outputs), dtype=np.float64)


def witness_mlp_regressor_r2_score(
    y_true: AbstractArray,
    y_pred: AbstractArray,
    *,
    sample_weight: AbstractArray | None = None,
) -> AbstractArray:
    """Describe the scalar MLPRegressor score helper output."""
    true_samples, true_outputs = _check_array(y_true, "y_true")
    pred_samples, pred_outputs = _check_array(y_pred, "y_pred")
    if true_samples != pred_samples or true_outputs != pred_outputs:
        raise ValueError("y_true and y_pred must have matching shapes")
    if sample_weight is not None:
        if len(sample_weight.shape) != 1 or int(sample_weight.shape[0]) != true_samples:
            raise ValueError("sample_weight must be 1D and match sample count")
    return AbstractArray(shape=(), dtype=np.float64)
