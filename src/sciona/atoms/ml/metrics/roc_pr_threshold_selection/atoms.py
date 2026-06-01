from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
import numpy as np
from numpy.typing import NDArray
import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_compute_roc_coordinates,
    witness_select_optimal_threshold_youden,
)

@register_atom(witness_compute_roc_coordinates, name="compute_roc_coordinates")
@icontract.require(lambda y_true, y_prob: len(y_true) == len(y_prob), "Precondition failed: len(y_true) == len(y_prob)")
@icontract.ensure(lambda result, y_true, y_prob: len(fpr) == len(thresholds), "Postcondition failed: len(fpr) == len(thresholds)")
def compute_roc_coordinates(y_true: NDArray[np.int64], y_prob: NDArray[np.float64]) -> NDArray[np.float64]:
    """Calculate FPR, TPR, and threshold sets.

    Args:
        y_true: NDArray[np.int64]
        y_prob: NDArray[np.float64]

    Returns:
        fpr: NDArray[np.float64]
    """
    import sklearn.metrics
    return sklearn.metrics.roc_curve(y_true=y_true, y_prob=y_prob) # type: ignore

@register_atom(witness_select_optimal_threshold_youden, name="select_optimal_threshold_youden")
@icontract.require(lambda fpr, tpr, thresholds: fpr is not None, "Precondition failed: fpr is not None")
@icontract.ensure(lambda result, fpr, tpr, thresholds: 0.0 <= optimal_threshold <= 1.0, "Postcondition failed: 0.0 <= optimal_threshold <= 1.0")
def select_optimal_threshold_youden(fpr: NDArray[np.float64], tpr: NDArray[np.float64], thresholds: NDArray[np.float64]) -> float:
    """Determine the optimal classification threshold by maximizing Youden's J statistic.

    Args:
        fpr: NDArray[np.float64]
        tpr: NDArray[np.float64]
        thresholds: NDArray[np.float64]

    Returns:
        optimal_threshold: float
    """
    import numpy
    return numpy.argmax(fpr=fpr, tpr=tpr, thresholds=thresholds) # type: ignore

