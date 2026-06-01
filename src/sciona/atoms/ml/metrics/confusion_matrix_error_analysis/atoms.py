from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
import numpy as np
from numpy.typing import NDArray
import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_build_raw_confusion_matrix,
    witness_normalize_confusion_matrix,
)

@register_atom(witness_build_raw_confusion_matrix, name="build_raw_confusion_matrix")
@icontract.require(lambda y_true, y_pred, labels: len(y_true) == len(y_pred), "Precondition failed: len(y_true) == len(y_pred)")
@icontract.ensure(lambda result, y_true, y_pred, labels: raw_matrix.shape[0] == len(labels), "Postcondition failed: raw_matrix.shape[0] == len(labels)")
@icontract.ensure(lambda result, y_true, y_pred, labels: raw_matrix.shape[1] == len(labels), "Postcondition failed: raw_matrix.shape[1] == len(labels)")
def build_raw_confusion_matrix(y_true: NDArray[Any], y_pred: NDArray[Any], labels: Any) -> NDArray[np.int64]:
    """Construct a 2D square matrix of prediction/target joint frequencies.

    Args:
        y_true: NDArray[Any]
        y_pred: NDArray[Any]
        labels: list[Any]

    Returns:
        raw_matrix: NDArray[np.int64]
    """
    import sklearn.metrics
    return sklearn.metrics.confusion_matrix(y_true=y_true, y_pred=y_pred, labels=labels) # type: ignore

@register_atom(witness_normalize_confusion_matrix, name="normalize_confusion_matrix")
@icontract.require(lambda raw_matrix, axis: raw_matrix is not None, "Precondition failed: raw_matrix is not None")
@icontract.ensure(lambda result, raw_matrix, axis: result is not None, "Postcondition failed: result is not None")
def normalize_confusion_matrix(raw_matrix: NDArray[np.int64], axis: str) -> NDArray[np.float64]:
    """Scale joint matrix by total counts along specified axis.

    Args:
        raw_matrix: NDArray[np.int64]
        axis: str

    Returns:
        normalized_matrix: NDArray[np.float64]
    """
    import numpy
    return numpy.divide(raw_matrix=raw_matrix, axis=axis) # type: ignore

