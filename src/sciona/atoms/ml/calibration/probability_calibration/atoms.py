from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
import numpy as np
from numpy.typing import NDArray
import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_compute_calibration_binned_frequencies,
    witness_fit_platt_scaling,
)

@register_atom(witness_compute_calibration_binned_frequencies, name="compute_calibration_binned_frequencies")
@icontract.require(lambda y_true, y_prob, n_bins: n_bins >= 2, "Precondition failed: n_bins >= 2")
@icontract.ensure(lambda result, y_true, y_prob, n_bins: len(true_ratios) == len(pred_probabilities), "Postcondition failed: len(true_ratios) == len(pred_probabilities)")
def compute_calibration_binned_frequencies(y_true: NDArray[np.int64], y_prob: NDArray[np.float64], n_bins: int) -> NDArray[np.float64]:
    """Group predictions into probability bins and calculate empirical true ratios.

    Args:
        y_true: NDArray[np.int64]
        y_prob: NDArray[np.float64]
        n_bins: int

    Returns:
        true_ratios: NDArray[np.float64]
    """
    import sklearn.calibration
    return sklearn.calibration.calibration_curve(y_true=y_true, y_prob=y_prob, n_bins=n_bins) # type: ignore

@register_atom(witness_fit_platt_scaling, name="fit_platt_scaling")
@icontract.require(lambda raw_scores, y_true: raw_scores is not None, "Precondition failed: raw_scores is not None")
@icontract.ensure(lambda result, raw_scores, y_true: result is not None, "Postcondition failed: result is not None")
def fit_platt_scaling(raw_scores: NDArray[np.float64], y_true: NDArray[np.int64]) -> Any:
    """Train logistic sigmoid mapping function over raw decision outputs.

    Args:
        raw_scores: NDArray[np.float64]
        y_true: NDArray[np.int64]

    Returns:
        calibrator_state: Any
    """
    import sklearn.calibration
    return sklearn.calibration.CalibratedClassifierCV(raw_scores=raw_scores, y_true=y_true) # type: ignore

