from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
from sciona.ghost.abstract import AbstractArray, AbstractScalar, AbstractSignal

def witness_compute_raw_residuals(y_true: AbstractArray, y_pred: AbstractArray) -> AbstractArray:
    """Ghost witness for compute_raw_residuals."""
    _ = (y_true, y_pred)
    return AbstractArray(shape=y_true.shape, dtype=y_true.dtype)

def witness_compute_l2_regression_metrics(y_true: AbstractArray, residuals: AbstractArray) -> AbstractScalar:
    """Ghost witness for compute_l2_regression_metrics."""
    _ = (y_true, residuals)
    return AbstractScalar(dtype="float64")

