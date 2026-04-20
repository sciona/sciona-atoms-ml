"""Covariance estimator helper atoms adapted from scikit-learn."""

from __future__ import annotations

import warnings

import icontract
import numpy as np
from numpy.typing import NDArray
from sklearn.utils import check_array

from sciona.ghost.registry import register_atom

from .witnesses import witness_empirical_covariance, witness_ledoit_wolf, witness_ledoit_wolf_shrinkage, witness_oas, witness_shrunk_covariance


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


@register_atom(witness_ledoit_wolf_shrinkage)
@icontract.require(lambda X: X.ndim == 2, "X must be 2D")
@icontract.require(lambda X: X.size > 0, "X must contain at least one value")
@icontract.require(lambda block_size: block_size >= 1, "block_size must be at least one")
@icontract.ensure(lambda result: np.isfinite(result), "shrinkage must be finite")
@icontract.ensure(lambda result: 0.0 <= result <= 1.0, "shrinkage must lie in [0, 1]")
def ledoit_wolf_shrinkage(
    X: NDArray[np.float64],
    *,
    assume_centered: bool = False,
    block_size: int = 1000,
) -> float:
    """Estimate the Ledoit-Wolf shrinkage coefficient from samples."""
    checked_x = check_array(X)
    if len(checked_x.shape) == 2 and checked_x.shape[1] == 1:
        return 0.0
    if checked_x.ndim == 1:
        checked_x = np.reshape(checked_x, (1, -1))
    if checked_x.shape[0] == 1:
        warnings.warn(
            "Only one sample available. You may want to reshape your data array",
            UserWarning,
            stacklevel=2,
        )

    n_samples, n_features = checked_x.shape
    if not assume_centered:
        checked_x = checked_x - checked_x.mean(0)

    n_splits = int(n_features / block_size)
    squared_x = checked_x**2
    emp_cov_trace = np.sum(squared_x, axis=0) / n_samples
    mu = np.sum(emp_cov_trace) / n_features
    beta_sum = 0.0
    delta_sum = 0.0

    for row_block in range(n_splits):
        rows = slice(block_size * row_block, block_size * (row_block + 1))
        for col_block in range(n_splits):
            cols = slice(block_size * col_block, block_size * (col_block + 1))
            beta_sum += np.sum(np.dot(squared_x.T[rows], squared_x[:, cols]))
            delta_sum += np.sum(np.dot(checked_x.T[rows], checked_x[:, cols]) ** 2)

        tail = slice(block_size * n_splits, None)
        beta_sum += np.sum(np.dot(squared_x.T[rows], squared_x[:, tail]))
        delta_sum += np.sum(np.dot(checked_x.T[rows], checked_x[:, tail]) ** 2)

    tail = slice(block_size * n_splits, None)
    for col_block in range(n_splits):
        cols = slice(block_size * col_block, block_size * (col_block + 1))
        beta_sum += np.sum(np.dot(squared_x.T[tail], squared_x[:, cols]))
        delta_sum += np.sum(np.dot(checked_x.T[tail], checked_x[:, cols]) ** 2)

    delta_sum += np.sum(np.dot(checked_x.T[tail], checked_x[:, tail]) ** 2)
    delta_sum /= n_samples**2
    beta_sum += np.sum(np.dot(squared_x.T[tail], squared_x[:, tail]))

    beta = 1.0 / (n_features * n_samples) * (beta_sum / n_samples - delta_sum)
    delta = delta_sum - 2.0 * mu * emp_cov_trace.sum() + n_features * mu**2
    delta /= n_features
    beta = min(beta, delta)
    return float(0.0 if beta == 0 else beta / delta)


@register_atom(witness_ledoit_wolf)
@icontract.require(lambda X: X.ndim == 2, "X must be 2D")
@icontract.require(lambda X: X.size > 0, "X must contain at least one value")
@icontract.require(lambda block_size: block_size >= 1, "block_size must be at least one")
@icontract.ensure(lambda result: result[0].ndim == 2, "covariance must be a matrix")
@icontract.ensure(lambda result: result[0].shape[0] == result[0].shape[1], "covariance must be square")
@icontract.ensure(lambda result: 0.0 <= result[1] <= 1.0, "shrinkage must lie in [0, 1]")
def ledoit_wolf(
    X: NDArray[np.float64],
    *,
    assume_centered: bool = False,
    block_size: int = 1000,
) -> tuple[NDArray[np.float64], float]:
    """Estimate a covariance matrix with the Ledoit-Wolf shrinkage rule."""
    checked_x = check_array(X)
    if len(checked_x.shape) == 2 and checked_x.shape[1] == 1:
        if not assume_centered:
            checked_x = checked_x - checked_x.mean()
        return np.atleast_2d((checked_x**2).mean()).astype(np.float64), 0.0

    shrinkage = ledoit_wolf_shrinkage(
        checked_x,
        assume_centered=assume_centered,
        block_size=block_size,
    )
    emp_cov = empirical_covariance(checked_x, assume_centered=assume_centered)
    return shrunk_covariance(emp_cov, shrinkage), shrinkage


@register_atom(witness_oas)
@icontract.require(lambda X: X.ndim == 2, "X must be 2D")
@icontract.require(lambda X: X.size > 0, "X must contain at least one value")
@icontract.ensure(lambda result: result[0].ndim == 2, "covariance must be a matrix")
@icontract.ensure(lambda result: result[0].shape[0] == result[0].shape[1], "covariance must be square")
@icontract.ensure(lambda result: 0.0 <= result[1] <= 1.0, "shrinkage must lie in [0, 1]")
def oas(
    X: NDArray[np.float64],
    *,
    assume_centered: bool = False,
) -> tuple[NDArray[np.float64], float]:
    """Estimate covariance with the Oracle Approximating Shrinkage formula."""
    checked_x = check_array(X)
    if len(checked_x.shape) == 2 and checked_x.shape[1] == 1:
        if not assume_centered:
            checked_x = checked_x - checked_x.mean()
        return np.atleast_2d((checked_x**2).mean()).astype(np.float64), 0.0

    n_samples, n_features = checked_x.shape
    emp_cov = empirical_covariance(checked_x, assume_centered=assume_centered)
    alpha = np.mean(emp_cov**2)
    mu = np.trace(emp_cov) / n_features
    mu_squared = mu**2
    numerator = alpha + mu_squared
    denominator = (n_samples + 1) * (alpha - mu_squared / n_features)
    shrinkage = 1.0 if denominator == 0 else min(numerator / denominator, 1.0)
    return shrunk_covariance(emp_cov, shrinkage), float(shrinkage)
