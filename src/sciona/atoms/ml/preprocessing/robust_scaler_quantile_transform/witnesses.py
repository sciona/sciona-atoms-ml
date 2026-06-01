from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
from sciona.ghost.abstract import AbstractArray, AbstractScalar, AbstractSignal

def witness_compute_column_percentiles(matrix: AbstractArray, percentiles: AbstractArray) -> AbstractArray:
    """Ghost witness for compute_column_percentiles."""
    _ = (matrix, percentiles)
    return AbstractArray(shape=matrix.shape, dtype=matrix.dtype)

def witness_apply_robust_normalization(matrix: AbstractArray, medians: AbstractArray, iqrs: AbstractArray) -> AbstractArray:
    """Ghost witness for apply_robust_normalization."""
    _ = (matrix, medians, iqrs)
    return AbstractArray(shape=matrix.shape, dtype=matrix.dtype)

