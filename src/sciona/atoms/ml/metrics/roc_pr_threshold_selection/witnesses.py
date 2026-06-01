from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
from sciona.ghost.abstract import AbstractArray, AbstractScalar, AbstractSignal

def witness_compute_roc_coordinates(y_true: AbstractArray, y_prob: AbstractArray) -> AbstractArray:
    """Ghost witness for compute_roc_coordinates."""
    _ = (y_true, y_prob)
    return AbstractArray(shape=y_true.shape, dtype=y_true.dtype)

def witness_select_optimal_threshold_youden(fpr: AbstractArray, tpr: AbstractArray, thresholds: AbstractArray) -> AbstractScalar:
    """Ghost witness for select_optimal_threshold_youden."""
    _ = (fpr, tpr, thresholds)
    return AbstractScalar(dtype="float64")

