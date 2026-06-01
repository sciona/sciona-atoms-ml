from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
from sciona.ghost.abstract import AbstractArray, AbstractScalar, AbstractSignal

def witness_build_raw_confusion_matrix(y_true: AbstractArray, y_pred: AbstractArray, labels: AbstractScalar | Any) -> AbstractArray:
    """Ghost witness for build_raw_confusion_matrix."""
    _ = (y_true, y_pred, labels)
    return AbstractArray(shape=y_true.shape, dtype=y_true.dtype)

def witness_normalize_confusion_matrix(raw_matrix: AbstractArray, axis: AbstractScalar | str) -> AbstractArray:
    """Ghost witness for normalize_confusion_matrix."""
    _ = (raw_matrix, axis)
    return AbstractArray(shape=raw_matrix.shape, dtype=raw_matrix.dtype)

