from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
from sciona.ghost.abstract import AbstractArray, AbstractScalar, AbstractSignal

def witness_compute_imputation_statistics(matrix: AbstractArray, strategy: AbstractScalar | str) -> AbstractArray:
    """Ghost witness for compute_imputation_statistics."""
    _ = (matrix, strategy)
    return AbstractArray(shape=matrix.shape, dtype=matrix.dtype)

def witness_fill_missing_values(matrix: AbstractArray, statistics: AbstractArray) -> AbstractArray:
    """Ghost witness for fill_missing_values."""
    _ = (matrix, statistics)
    return AbstractArray(shape=matrix.shape, dtype=matrix.dtype)

