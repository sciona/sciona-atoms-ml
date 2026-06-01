from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
from sciona.ghost.abstract import AbstractArray, AbstractScalar, AbstractSignal

def witness_compute_calibration_binned_frequencies(y_true: AbstractArray, y_prob: AbstractArray, n_bins: AbstractScalar | int) -> AbstractArray:
    """Ghost witness for compute_calibration_binned_frequencies."""
    _ = (y_true, y_prob, n_bins)
    return AbstractArray(shape=y_true.shape, dtype=y_true.dtype)

def witness_fit_platt_scaling(raw_scores: AbstractArray, y_true: AbstractArray) -> AbstractScalar:
    """Ghost witness for fit_platt_scaling."""
    _ = (raw_scores, y_true)
    return AbstractScalar(dtype="float64")

