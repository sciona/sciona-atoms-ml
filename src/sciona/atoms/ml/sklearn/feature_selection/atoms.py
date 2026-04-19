"""Univariate feature scoring atoms adapted from scikit-learn."""

from __future__ import annotations

import warnings

import icontract
import numpy as np
import scipy.stats as stats
from numpy.typing import NDArray
from scipy import special
from scipy.sparse import issparse
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import as_float_array, check_X_y, safe_mask, safe_sqr
from sklearn.utils.extmath import row_norms, safe_sparse_dot

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_chi2,
    witness_f_classif,
    witness_f_regression,
    witness_r_regression,
)

ScoreResult = tuple[NDArray[np.float64], NDArray[np.float64]]


def _f_oneway(*arrays: NDArray[np.float64]) -> ScoreResult:
    n_classes = len(arrays)
    class_arrays = [as_float_array(array) for array in arrays]
    n_samples_per_class = np.array([array.shape[0] for array in class_arrays])
    n_samples = np.sum(n_samples_per_class)
    ss_alldata = sum(safe_sqr(array).sum(axis=0) for array in class_arrays)
    sums_args = [np.asarray(array.sum(axis=0)) for array in class_arrays]
    square_of_sums_alldata = sum(sums_args) ** 2
    square_of_sums_args = [value**2 for value in sums_args]
    sstot = ss_alldata - square_of_sums_alldata / float(n_samples)
    ssbn = 0.0
    for class_index, _array in enumerate(class_arrays):
        ssbn += square_of_sums_args[class_index] / n_samples_per_class[class_index]
    ssbn -= square_of_sums_alldata / float(n_samples)
    sswn = sstot - ssbn
    dfbn = n_classes - 1
    dfwn = n_samples - n_classes
    msb = ssbn / float(dfbn)
    msw = sswn / float(dfwn)
    constant_features_idx = np.where(msw == 0.0)[0]
    if np.nonzero(msb)[0].size != msb.size and constant_features_idx.size:
        warnings.warn(f"Features {constant_features_idx} are constant.", UserWarning, stacklevel=2)
    f_statistic = np.asarray(msb / msw).ravel()
    p_values = special.fdtrc(dfbn, dfwn, f_statistic)
    return np.asarray(f_statistic, dtype=np.float64), np.asarray(p_values, dtype=np.float64)


def _chisquare(
    observed: NDArray[np.float64],
    expected: NDArray[np.float64],
) -> ScoreResult:
    observed_float = np.asarray(observed, dtype=np.float64)
    k = len(observed_float)
    chisq = observed_float
    chisq -= expected
    chisq **= 2
    with np.errstate(invalid="ignore"):
        chisq /= expected
    chi2_stats = chisq.sum(axis=0)
    p_values = special.chdtrc(k - 1, chi2_stats)
    return np.asarray(chi2_stats, dtype=np.float64), np.asarray(p_values, dtype=np.float64)


def _finite_p_values(result: ScoreResult) -> bool:
    p_values = result[1]
    finite = np.isfinite(p_values)
    return bool(np.all((p_values[finite] >= 0.0) & (p_values[finite] <= 1.0)))


@register_atom(witness_f_classif)
@icontract.require(lambda X: X.ndim == 2, "X must be 2D")
@icontract.require(lambda X, y: X.shape[0] == y.shape[0], "X and y must have equal sample count")
@icontract.require(lambda y: np.unique(y).size >= 2, "y must contain at least two classes")
@icontract.require(lambda X, y: X.shape[0] > np.unique(y).size, "residual degrees of freedom must be positive")
@icontract.ensure(lambda result, X: result[0].shape == (X.shape[1],), "F statistics must match feature count")
@icontract.ensure(lambda result, X: result[1].shape == (X.shape[1],), "p-values must match feature count")
@icontract.ensure(lambda result: _finite_p_values(result), "finite p-values must be probabilities")
def f_classif(X: NDArray[np.float64], y: NDArray[np.float64]) -> ScoreResult:
    """Compute one-way ANOVA F statistics for each feature by class label."""
    checked_x, checked_y = check_X_y(X, y, accept_sparse=["csr", "csc", "coo"])
    class_arrays = [checked_x[safe_mask(checked_x, checked_y == label)] for label in np.unique(checked_y)]
    return _f_oneway(*class_arrays)


@register_atom(witness_chi2)
@icontract.require(lambda X: X.ndim == 2, "X must be 2D")
@icontract.require(lambda X, y: X.shape[0] == y.shape[0], "X and y must have equal sample count")
@icontract.require(lambda X: bool(np.all((X.data if issparse(X) else X) >= 0)), "X must be non-negative")
@icontract.require(lambda y: np.unique(y).size >= 2, "y must contain at least two classes")
@icontract.ensure(lambda result, X: result[0].shape == (X.shape[1],), "chi-square statistics must match feature count")
@icontract.ensure(lambda result, X: result[1].shape == (X.shape[1],), "p-values must match feature count")
@icontract.ensure(lambda result: _finite_p_values(result), "finite p-values must be probabilities")
def chi2(X: NDArray[np.float64], y: NDArray[np.float64]) -> ScoreResult:
    """Compute chi-square dependence statistics between features and classes."""
    checked_x = check_X_y(X, y, accept_sparse="csr", dtype=(np.float64, np.float32))[0]
    if np.any((checked_x.data if issparse(checked_x) else checked_x) < 0):
        raise ValueError("Input X must be non-negative.")

    y_matrix = LabelBinarizer(sparse_output=True).fit_transform(y)
    if y_matrix.shape[1] == 1:
        y_matrix = y_matrix.toarray()
        y_matrix = np.append(1 - y_matrix, y_matrix, axis=1)

    observed = safe_sparse_dot(y_matrix.T, checked_x)
    if issparse(observed):
        observed = observed.toarray()
    feature_count = checked_x.sum(axis=0).reshape(1, -1)
    class_prob = y_matrix.mean(axis=0).reshape(1, -1)
    expected = np.dot(class_prob.T, feature_count)
    return _chisquare(np.asarray(observed, dtype=np.float64), np.asarray(expected, dtype=np.float64))


@register_atom(witness_r_regression)
@icontract.require(lambda X: X.ndim == 2, "X must be 2D")
@icontract.require(lambda X, y: X.shape[0] == y.shape[0], "X and y must have equal sample count")
@icontract.require(lambda X: X.shape[0] >= 2, "need at least two samples")
@icontract.ensure(lambda result, X: result.shape == (X.shape[1],), "correlations must match feature count")
@icontract.ensure(lambda result: np.all(np.abs(result[np.isfinite(result)]) <= 1.0 + 1e-12), "finite correlations must lie in [-1, 1]")
def r_regression(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    *,
    center: bool = True,
    force_finite: bool = True,
) -> NDArray[np.float64]:
    """Compute Pearson correlation between each feature and a numeric target."""
    checked_x, checked_y = check_X_y(X, y, accept_sparse=["csr", "csc", "coo"], dtype=np.float64)
    n_samples = checked_x.shape[0]

    if center:
        centered_y = checked_y - np.mean(checked_y)
        x_means = checked_x.mean(axis=0)
        x_means = x_means.getA1() if isinstance(x_means, np.matrix) else x_means
        x_norms = np.sqrt(row_norms(checked_x.T, squared=True) - n_samples * x_means**2)
    else:
        centered_y = checked_y
        x_norms = row_norms(checked_x.T)

    correlation_coefficient = safe_sparse_dot(centered_y, checked_x)
    with np.errstate(divide="ignore", invalid="ignore"):
        correlation_coefficient /= x_norms
        correlation_coefficient /= np.linalg.norm(centered_y)

    result = np.asarray(correlation_coefficient, dtype=np.float64)
    if force_finite and not np.isfinite(result).all():
        nan_mask = np.isnan(result)
        result[nan_mask] = 0.0
    return result


@register_atom(witness_f_regression)
@icontract.require(lambda X: X.ndim == 2, "X must be 2D")
@icontract.require(lambda X, y: X.shape[0] == y.shape[0], "X and y must have equal sample count")
@icontract.require(lambda X, center: X.shape[0] > (2 if center else 1), "degrees of freedom must be positive")
@icontract.ensure(lambda result, X: result[0].shape == (X.shape[1],), "F statistics must match feature count")
@icontract.ensure(lambda result, X: result[1].shape == (X.shape[1],), "p-values must match feature count")
@icontract.ensure(lambda result: _finite_p_values(result), "finite p-values must be probabilities")
def f_regression(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    *,
    center: bool = True,
    force_finite: bool = True,
) -> ScoreResult:
    """Convert feature-target correlations into regression F statistics."""
    correlation_coefficient = r_regression(
        X,
        y,
        center=center,
        force_finite=force_finite,
    )
    deg_of_freedom = y.size - (2 if center else 1)
    corr_coef_squared = correlation_coefficient**2

    with np.errstate(divide="ignore", invalid="ignore"):
        f_statistic = corr_coef_squared / (1 - corr_coef_squared) * deg_of_freedom
        p_values = stats.f.sf(f_statistic, 1, deg_of_freedom)

    f_result = np.asarray(f_statistic, dtype=np.float64)
    p_result = np.asarray(p_values, dtype=np.float64)
    if force_finite and not np.isfinite(f_result).all():
        mask_inf = np.isinf(f_result)
        f_result[mask_inf] = np.finfo(f_result.dtype).max
        p_result[mask_inf] = 0.0
        mask_nan = np.isnan(f_result)
        f_result[mask_nan] = 0.0
        p_result[mask_nan] = 1.0
    return f_result, p_result
