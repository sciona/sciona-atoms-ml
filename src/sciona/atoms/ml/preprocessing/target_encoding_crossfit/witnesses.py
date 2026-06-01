from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
from sciona.ghost.abstract import AbstractArray, AbstractScalar, AbstractSignal

def witness_compute_global_prior(targets: AbstractArray) -> AbstractScalar:
    """Ghost witness for compute_global_prior."""
    _ = (targets)
    return AbstractScalar(dtype="float64")

def witness_compute_oof_target_means(categorical_column: AbstractArray, targets: AbstractArray, prior: AbstractScalar | float, smoothing: AbstractScalar | float, train_indices: AbstractArray, test_indices: AbstractArray) -> AbstractArray:
    """Ghost witness for compute_oof_target_means."""
    _ = (categorical_column, targets, prior, smoothing, train_indices, test_indices)
    return AbstractArray(shape=categorical_column.shape, dtype=categorical_column.dtype)

