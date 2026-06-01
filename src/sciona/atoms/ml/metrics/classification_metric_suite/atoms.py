from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
import numpy as np
from numpy.typing import NDArray
import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_compute_binary_classification_stats,
    witness_compute_f_beta_score,
)

@register_atom(witness_compute_binary_classification_stats, name="compute_binary_classification_stats")
@icontract.require(lambda y_true, y_pred: len(y_true) == len(y_pred), "Precondition failed: len(y_true) == len(y_pred)")
@icontract.ensure(lambda result, y_true, y_pred: result is not None, "Postcondition failed: result is not None")
def compute_binary_classification_stats(y_true: NDArray[np.int64], y_pred: NDArray[np.int64]) -> int:
    """Calculate true/false positive/negative counts from binary label vectors.

    Args:
        y_true: binary array
        y_pred: binary array

    Returns:
        tp: int
    """
    import sklearn.metrics
    return sklearn.metrics.confusion_matrix(y_true=y_true, y_pred=y_pred) # type: ignore

@register_atom(witness_compute_f_beta_score, name="compute_f_beta_score")
@icontract.require(lambda tp, fp, tn, fn, beta: beta > 0.0, "Precondition failed: beta > 0.0")
@icontract.ensure(lambda result, tp, fp, tn, fn, beta: 0.0 <= precision <= 1.0, "Postcondition failed: 0.0 <= precision <= 1.0")
def compute_f_beta_score(tp: int, fp: int, tn: int, fn: int, beta: float) -> float:
    """Calculate Precision, Recall, and F-beta metrics based on prediction counts.

    Args:
        tp: int
        fp: int
        tn: int
        fn: int
        beta: float

    Returns:
        precision: float
    """
    import sklearn.metrics
    return sklearn.metrics.fbeta_score(tp=tp, fp=fp, tn=tn, fn=fn, beta=beta) # type: ignore

