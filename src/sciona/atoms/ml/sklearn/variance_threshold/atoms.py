"""VarianceThreshold estimator atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .state_models import VarianceThresholdState
from .witnesses import (
    witness_variance_threshold_fit,
    witness_variance_threshold_support_mask,
    witness_variance_threshold_transform,
)

MatrixLike = NDArray[np.float64] | sp.spmatrix

def _is_2d(X: MatrixLike) -> bool:
    return bool(getattr(X, "ndim", 0) == 2)

def _feature_count(X: MatrixLike) -> int:
    return int(X.shape[1])

def _row_count(X: MatrixLike) -> int:
    return int(X.shape[0])

@register_atom(witness_variance_threshold_fit)
@icontract.require(lambda X: _is_2d(X), "X must be a 2D matrix")
@icontract.require(lambda X: _row_count(X) >= 1, "X must contain at least one sample")
@icontract.require(lambda threshold: threshold >= 0.0, "threshold must be non-negative")
@icontract.ensure(lambda result, X: result.variances.shape == (_feature_count(X),), "one variance per feature")
@icontract.ensure(lambda result: result.n_features_in == result.variances.shape[0], "state feature count must match variances")
@icontract.ensure(lambda result: result.threshold >= 0.0, "state threshold must be non-negative")
def variance_threshold_fit(
    X: MatrixLike,
    threshold: float = 0.0,
) -> VarianceThresholdState:
    from sklearn.utils import check_array
    from sklearn.utils.sparsefuncs import mean_variance_axis, min_max_axis
    """Learn per-feature variances for a VarianceThreshold selector."""
    checked_x = check_array(
        X,
        accept_sparse=("csr", "csc"),
        dtype=np.float64,
        ensure_all_finite="allow-nan",
    )

    if hasattr(checked_x, "toarray"):
        _means, variances = mean_variance_axis(checked_x, axis=0)
        if threshold == 0.0:
            mins, maxes = min_max_axis(checked_x, axis=0)
            peak_to_peaks = maxes - mins
    else:
        variances = np.nanvar(checked_x, axis=0)
        if threshold == 0.0:
            peak_to_peaks = np.ptp(checked_x, axis=0)

    variance_array = np.asarray(variances, dtype=np.float64)
    if threshold == 0.0:
        compare_arr = np.array([variance_array, np.asarray(peak_to_peaks, dtype=np.float64)])
        variance_array = np.nanmin(compare_arr, axis=0)

    if np.all(~np.isfinite(variance_array) | (variance_array <= threshold)):
        message = f"No feature in X meets the variance threshold {threshold:.5f}"
        if checked_x.shape[0] == 1:
            message += " (X contains only one sample)"
        raise ValueError(message)

    return VarianceThresholdState(
        variances=variance_array,
        threshold=float(threshold),
        n_features_in=int(checked_x.shape[1]),
    )

@register_atom(witness_variance_threshold_support_mask)
@icontract.require(lambda state: state.n_features_in == state.variances.shape[0], "state feature count must match variances")
@icontract.ensure(lambda result, state: result.shape == (state.n_features_in,), "mask must match fitted feature count")
@icontract.ensure(lambda result: result.dtype == np.bool_, "support mask must be boolean")
def variance_threshold_support_mask(
    state: VarianceThresholdState,
) -> NDArray[np.bool_]:
    """Return the fitted feature mask whose variances exceed the threshold."""
    return np.asarray(state.variances > state.threshold, dtype=np.bool_)

@register_atom(witness_variance_threshold_transform)
@icontract.require(lambda X: _is_2d(X), "X must be a 2D matrix")
@icontract.require(lambda X, state: _feature_count(X) == state.n_features_in, "X feature count must match fitted state")
@icontract.ensure(lambda result, state: result.shape[1] == int(np.sum(state.variances > state.threshold)), "output columns must match selected features")
def variance_threshold_transform(
    X: MatrixLike,
    state: VarianceThresholdState,
) -> MatrixLike:
    from sklearn.utils import check_array
    """Select columns whose fitted variances exceed the threshold."""
    checked_x = check_array(
        X,
        accept_sparse=("csr", "csc"),
        dtype=np.float64,
        ensure_all_finite="allow-nan",
    )
    if checked_x.shape[1] != state.n_features_in:
        raise ValueError("X feature count does not match fitted state")
    support = variance_threshold_support_mask(state)
    return checked_x[:, support]
