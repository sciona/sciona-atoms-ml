"""Covariance estimator helper atoms adapted from scikit-learn."""

from __future__ import annotations

import warnings

import icontract
import numpy as np
from numpy.typing import NDArray
from sklearn.utils import check_array

from sciona.ghost.registry import register_atom

from .witnesses import witness_empirical_covariance, witness_shrunk_covariance


@register_atom(witness_empirical_covariance)
@icontract.require(lambda X: X.ndim in {1, 2}, "X must be 1D or 2D")
@icontract.require(lambda X: X.size > 0, "X must contain at least one value")
@icontract.ensure(lambda result: result.ndim == 2, "covariance must be a matrix")
@icontract.ensure(lambda result: result.shape[0] == result.shape[1], "covariance must be square")
@icontract.ensure(lambda result: np.allclose(result, result.T, equal_nan=True), "covariance must be symmetric")
def empirical_covariance(
    X: NDArray[np.float64],
    *,
    assume_centered: bool = False,
) -> NDArray[np.float64]:
    """Compute the maximum-likelihood empirical covariance matrix."""
    checked_x = check_array(X, ensure_2d=False, ensure_all_finite=False)
    if checked_x.ndim == 1:
        checked_x = np.reshape(checked_x, (1, -1))
    if checked_x.shape[0] == 1:
        warnings.warn(
            "Only one sample available. You may want to reshape your data array",
            UserWarning,
            stacklevel=2,
        )
    if assume_centered:
        covariance = np.dot(checked_x.T, checked_x) / checked_x.shape[0]
    else:
        covariance = np.cov(checked_x.T, bias=1)
    if covariance.ndim == 0:
        covariance = np.array([[covariance]])
    return np.asarray(covariance, dtype=np.float64)


@register_atom(witness_shrunk_covariance)
@icontract.require(lambda emp_cov: emp_cov.ndim >= 2, "emp_cov must be at least 2D")
@icontract.require(lambda emp_cov: emp_cov.shape[-1] == emp_cov.shape[-2], "last two dimensions must be square")
@icontract.require(lambda shrinkage: 0.0 <= shrinkage <= 1.0, "shrinkage must lie in [0, 1]")
@icontract.ensure(lambda result, emp_cov: result.shape == emp_cov.shape, "shrunk covariance must preserve shape")
@icontract.ensure(lambda result: np.all(np.isfinite(result)), "shrunk covariance must be finite")
def shrunk_covariance(
    emp_cov: NDArray[np.float64],
    shrinkage: float = 0.1,
) -> NDArray[np.float64]:
    """Shrink covariance matrices toward their average diagonal variance."""
    checked_cov = check_array(emp_cov, allow_nd=True)
    n_features = checked_cov.shape[-1]
    shrunk_cov = (1.0 - shrinkage) * checked_cov
    mu = np.trace(checked_cov, axis1=-2, axis2=-1) / n_features
    mu = np.expand_dims(mu, axis=tuple(range(mu.ndim, checked_cov.ndim)))
    shrunk_cov += shrinkage * mu * np.eye(n_features)
    return np.asarray(shrunk_cov, dtype=np.float64)
