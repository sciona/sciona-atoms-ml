"""Selected calibration atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray
from sklearn.utils import check_consistent_length, column_or_1d
from sklearn.utils.validation import _check_pos_label_consistency

from sciona.ghost.registry import register_atom

from .witnesses import witness_calibration_curve

CurveResult = tuple[NDArray[np.float64], NDArray[np.float64]]


def _is_1d(x: NDArray[np.float64]) -> bool:
    return bool(np.asarray(x).ndim == 1)


def _same_length(y_true: NDArray[np.float64], y_prob: NDArray[np.float64]) -> bool:
    return int(np.asarray(y_true).shape[0]) == int(np.asarray(y_prob).shape[0])


def _valid_strategy(strategy: str) -> bool:
    return strategy in {"uniform", "quantile"}


def _probabilities_in_unit_interval(y_prob: NDArray[np.float64]) -> bool:
    values = np.asarray(y_prob)
    return bool(values.size > 0 and values.min() >= 0.0 and values.max() <= 1.0)


def _curve_result_valid(result: CurveResult, n_bins: int) -> bool:
    prob_true, prob_pred = result
    if prob_true.shape != prob_pred.shape or prob_true.ndim != 1:
        return False
    if prob_true.shape[0] > n_bins:
        return False
    return bool(
        np.all((prob_true >= 0.0) & (prob_true <= 1.0))
        and np.all((prob_pred >= 0.0) & (prob_pred <= 1.0))
    )


@register_atom(witness_calibration_curve)
@icontract.require(lambda y_true: _is_1d(y_true), "y_true must be a 1D vector")
@icontract.require(lambda y_prob: _is_1d(y_prob), "y_prob must be a 1D vector")
@icontract.require(lambda y_true, y_prob: _same_length(y_true, y_prob), "y_true and y_prob must have equal sample count")
@icontract.require(lambda y_prob: _probabilities_in_unit_interval(y_prob), "y_prob values must be in [0, 1]")
@icontract.require(lambda n_bins: n_bins >= 1, "n_bins must be at least 1")
@icontract.require(lambda strategy: _valid_strategy(strategy), "strategy must be 'uniform' or 'quantile'")
@icontract.ensure(lambda result, n_bins: _curve_result_valid(result, n_bins), "curve outputs must be probability vectors with at most n_bins entries")
def calibration_curve(
    y_true: NDArray[np.float64],
    y_prob: NDArray[np.float64],
    *,
    pos_label: int | float | bool | str | None = None,
    n_bins: int = 5,
    strategy: str = "uniform",
) -> CurveResult:
    """Compute positive rate and mean predicted probability by calibration bin."""
    checked_y_true = column_or_1d(y_true)
    checked_y_prob = column_or_1d(y_prob)
    check_consistent_length(checked_y_true, checked_y_prob)
    checked_pos_label = _check_pos_label_consistency(pos_label, checked_y_true)

    if checked_y_prob.min() < 0 or checked_y_prob.max() > 1:
        raise ValueError("y_prob has values outside [0, 1].")

    labels = np.unique(checked_y_true)
    if len(labels) > 2:
        raise ValueError(f"Only binary classification is supported. Provided labels {labels}.")
    positive_mask = checked_y_true == checked_pos_label

    if strategy == "quantile":
        quantiles = np.linspace(0, 1, n_bins + 1)
        bins = np.percentile(checked_y_prob, quantiles * 100)
    elif strategy == "uniform":
        bins = np.linspace(0.0, 1.0, n_bins + 1)
    else:
        raise ValueError("Invalid entry to 'strategy' input. Strategy must be either 'quantile' or 'uniform'.")

    binids = np.searchsorted(bins[1:-1], checked_y_prob)
    bin_sums = np.bincount(binids, weights=checked_y_prob, minlength=len(bins))
    bin_true = np.bincount(binids, weights=positive_mask, minlength=len(bins))
    bin_total = np.bincount(binids, minlength=len(bins))

    nonzero = bin_total != 0
    prob_true = bin_true[nonzero] / bin_total[nonzero]
    prob_pred = bin_sums[nonzero] / bin_total[nonzero]
    return np.asarray(prob_true, dtype=np.float64), np.asarray(prob_pred, dtype=np.float64)
