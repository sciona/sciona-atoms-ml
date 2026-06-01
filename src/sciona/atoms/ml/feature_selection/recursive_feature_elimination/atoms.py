from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
import numpy as np
from numpy.typing import NDArray
import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_execute_rfe_step,
)

@register_atom(witness_execute_rfe_step, name="execute_rfe_step")
@icontract.require(lambda features, targets, active_mask, estimator, step_size: features.shape[1] == len(active_mask), "Precondition failed: features.shape[1] == len(active_mask)")
@icontract.ensure(lambda result, features, targets, active_mask, estimator, step_size: np.sum(updated_mask) < np.sum(active_mask), "Postcondition failed: np.sum(updated_mask) < np.sum(active_mask)")
def execute_rfe_step(features: NDArray[np.float64], targets: NDArray[Any], active_mask: NDArray[np.bool_], estimator: Any, step_size: int) -> NDArray[np.bool_]:
    """Train estimator, compute coefficients/importances, and drop the lowest scoring features.

    Args:
        features: NDArray[np.float64]
        targets: NDArray[Any]
        active_mask: NDArray[np.bool_]
        estimator: Any
        step_size: int

    Returns:
        updated_mask: NDArray[np.bool_]
    """
    import sklearn.feature_selection
    return sklearn.feature_selection.RFE(features=features, targets=targets, active_mask=active_mask, estimator=estimator, step_size=step_size) # type: ignore

