from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
from sciona.ghost.abstract import AbstractArray, AbstractScalar, AbstractSignal

def witness_compute_binary_classification_stats(y_true: AbstractArray, y_pred: AbstractArray) -> AbstractScalar:
    """Ghost witness for compute_binary_classification_stats."""
    _ = (y_true, y_pred)
    return AbstractScalar(dtype="float64")

def witness_compute_f_beta_score(tp: AbstractScalar | int, fp: AbstractScalar | int, tn: AbstractScalar | int, fn: AbstractScalar | int, beta: AbstractScalar | float) -> AbstractScalar:
    """Ghost witness for compute_f_beta_score."""
    _ = (tp, fp, tn, fn, beta)
    return AbstractScalar(dtype="float64")

