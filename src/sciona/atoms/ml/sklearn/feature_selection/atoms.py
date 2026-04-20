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
from sklearn.utils import as_float_array, check_array, check_X_y, safe_mask, safe_sqr
from sklearn.utils.extmath import row_norms, safe_sparse_dot

from sciona.ghost.registry import register_atom

from .state_models import UnivariateSelectionState
from .witnesses import (
    witness_chi2,
    witness_f_classif,
    witness_f_regression,
    witness_generic_univariate_select_fit,
    witness_r_regression,
    witness_select_fdr_fit,
    witness_select_fpr_fit,
    witness_select_fwe_fit,
    witness_select_k_best_fit,
    witness_select_percentile_fit,
    witness_univariate_selection_transform,
)

ScoreResult = tuple[NDArray[np.float64], NDArray[np.float64]]

SupportedScoreFunc = str
SelectorParam = int | float | str


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


def _selector_score_func_valid(score_func: SupportedScoreFunc) -> bool:
    return score_func in {"f_classif", "chi2", "f_regression"}


def _selector_scores(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    score_func: SupportedScoreFunc,
) -> ScoreResult:
    if score_func == "f_classif":
        return f_classif(X, y)
    if score_func == "chi2":
        return chi2(X, y)
    if score_func == "f_regression":
        return f_regression(X, y)
    raise ValueError(f"unsupported score_func: {score_func!r}")


def _clean_nans(scores: NDArray[np.float64]) -> NDArray[np.float64]:
    cleaned = as_float_array(scores, copy=True)
    cleaned[np.isnan(cleaned)] = np.finfo(cleaned.dtype).min
    return np.asarray(cleaned, dtype=np.float64)


def _k_best_valid(k: int | str, X: NDArray[np.float64]) -> bool:
    del X
    return bool(k == "all" or (isinstance(k, int) and not isinstance(k, bool) and k >= 0))


def _percentile_valid(percentile: float) -> bool:
    return bool(isinstance(percentile, (int, float)) and not isinstance(percentile, bool) and 0.0 <= float(percentile) <= 100.0)


def _alpha_valid(alpha: float) -> bool:
    return bool(isinstance(alpha, (int, float)) and not isinstance(alpha, bool) and 0.0 <= float(alpha) <= 1.0)


def _generic_mode_param_valid(mode: str, param: SelectorParam) -> bool:
    if mode == "k_best":
        return bool(param == "all" or (isinstance(param, int) and not isinstance(param, bool) and param >= 0))
    if mode == "percentile":
        return _percentile_valid(float(param)) if isinstance(param, (int, float)) and not isinstance(param, bool) else False
    if mode in {"fpr", "fdr", "fwe"}:
        return _alpha_valid(float(param)) if isinstance(param, (int, float)) and not isinstance(param, bool) else False
    return False


def _support_k_best(scores: NDArray[np.float64], k: int | str) -> NDArray[np.bool_]:
    if k == "all":
        return np.ones(scores.shape, dtype=bool)
    if k == 0:
        return np.zeros(scores.shape, dtype=bool)
    clean_scores = _clean_nans(scores)
    mask = np.zeros(clean_scores.shape, dtype=bool)
    mask[np.argsort(clean_scores, kind="mergesort")[-int(k) :]] = True
    return np.asarray(mask, dtype=np.bool_)


def _support_percentile(scores: NDArray[np.float64], percentile: float) -> NDArray[np.bool_]:
    if percentile == 100:
        return np.ones(len(scores), dtype=bool)
    if percentile == 0:
        return np.zeros(len(scores), dtype=bool)
    clean_scores = _clean_nans(scores)
    threshold = np.percentile(clean_scores, 100.0 - float(percentile))
    mask = clean_scores > threshold
    ties = np.where(clean_scores == threshold)[0]
    if len(ties):
        max_features = int(len(clean_scores) * float(percentile) / 100.0)
        kept_ties = ties[: max_features - int(mask.sum())]
        mask[kept_ties] = True
    return np.asarray(mask, dtype=np.bool_)


def _support_fpr(pvalues: NDArray[np.float64], alpha: float) -> NDArray[np.bool_]:
    return np.asarray(pvalues < float(alpha), dtype=np.bool_)


def _support_fdr(pvalues: NDArray[np.float64], alpha: float) -> NDArray[np.bool_]:
    n_features = len(pvalues)
    sorted_pvalues = np.sort(pvalues)
    selected = sorted_pvalues[sorted_pvalues <= float(alpha) / n_features * np.arange(1, n_features + 1)]
    if selected.size == 0:
        return np.zeros_like(pvalues, dtype=bool)
    return np.asarray(pvalues <= selected.max(), dtype=np.bool_)


def _support_fwe(pvalues: NDArray[np.float64], alpha: float) -> NDArray[np.bool_]:
    return np.asarray(pvalues < float(alpha) / len(pvalues), dtype=np.bool_)


def _selection_state_valid(state: UnivariateSelectionState) -> bool:
    pvalues_valid = state.pvalues is None or (
        state.pvalues.shape == (state.n_features_in,)
        and np.all(np.isfinite(state.pvalues))
        and np.all((state.pvalues >= 0.0) & (state.pvalues <= 1.0))
    )
    return bool(
        state.scores.shape == (state.n_features_in,)
        and state.support_mask.shape == (state.n_features_in,)
        and state.support_mask.dtype == np.bool_
        and state.n_features_in >= 1
        and _selector_score_func_valid(state.score_func)
        and state.selector in {"k_best", "percentile", "fpr", "fdr", "fwe", "generic"}
        and not np.any(np.isnan(state.scores))
        and pvalues_valid
    )


def _selection_transform_valid(result: NDArray[np.float64], X: NDArray[np.float64], state: UnivariateSelectionState) -> bool:
    values = np.asarray(result)
    return bool(values.ndim == 2 and values.shape == (np.asarray(X).shape[0], int(state.support_mask.sum())))


def _feature_count_matches(X: NDArray[np.float64], state: UnivariateSelectionState) -> bool:
    return bool(np.asarray(X).ndim == 2 and np.asarray(X).shape[1] == state.n_features_in)


def _build_selection_state(
    X: NDArray[np.float64],
    scores: NDArray[np.float64],
    pvalues: NDArray[np.float64],
    *,
    score_func: SupportedScoreFunc,
    selector: str,
    selector_param: SelectorParam,
    support_mask: NDArray[np.bool_],
) -> UnivariateSelectionState:
    return UnivariateSelectionState(
        scores=np.asarray(scores, dtype=np.float64),
        pvalues=np.asarray(pvalues, dtype=np.float64),
        support_mask=np.asarray(support_mask, dtype=np.bool_),
        n_features_in=int(np.asarray(X).shape[1]),
        score_func=score_func,
        selector=selector,
        selector_param=selector_param,
    )


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


@register_atom(witness_select_k_best_fit)
@icontract.require(lambda X: X.ndim == 2, "X must be 2D")
@icontract.require(lambda X, y: X.shape[0] == y.shape[0], "X and y must have equal sample count")
@icontract.require(lambda score_func: _selector_score_func_valid(score_func), "score_func must be one of the supported scoring atoms")
@icontract.require(lambda k, X: _k_best_valid(k, X), "k must be a non-negative integer or 'all'")
@icontract.ensure(lambda result: _selection_state_valid(result), "selection state must contain scores and a boolean support mask")
def select_k_best_fit(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    *,
    score_func: SupportedScoreFunc = "f_classif",
    k: int | str = 10,
) -> UnivariateSelectionState:
    """Fit a selector that keeps the k highest univariate feature scores."""
    checked_x, checked_y = check_X_y(X, y, accept_sparse=False, dtype=np.float64)
    scores, pvalues = _selector_scores(checked_x, checked_y, score_func)
    support_mask = _support_k_best(scores, k)
    return _build_selection_state(
        checked_x,
        scores,
        pvalues,
        score_func=score_func,
        selector="k_best",
        selector_param=k,
        support_mask=support_mask,
    )


@register_atom(witness_select_percentile_fit)
@icontract.require(lambda X: X.ndim == 2, "X must be 2D")
@icontract.require(lambda X, y: X.shape[0] == y.shape[0], "X and y must have equal sample count")
@icontract.require(lambda score_func: _selector_score_func_valid(score_func), "score_func must be one of the supported scoring atoms")
@icontract.require(lambda percentile: _percentile_valid(percentile), "percentile must lie in [0, 100]")
@icontract.ensure(lambda result: _selection_state_valid(result), "selection state must contain scores and a boolean support mask")
def select_percentile_fit(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    *,
    score_func: SupportedScoreFunc = "f_classif",
    percentile: float = 10.0,
) -> UnivariateSelectionState:
    """Fit a selector that keeps a percentile of the highest feature scores."""
    checked_x, checked_y = check_X_y(X, y, accept_sparse=False, dtype=np.float64)
    scores, pvalues = _selector_scores(checked_x, checked_y, score_func)
    support_mask = _support_percentile(scores, float(percentile))
    return _build_selection_state(
        checked_x,
        scores,
        pvalues,
        score_func=score_func,
        selector="percentile",
        selector_param=float(percentile),
        support_mask=support_mask,
    )


@register_atom(witness_select_fpr_fit)
@icontract.require(lambda X: X.ndim == 2, "X must be 2D")
@icontract.require(lambda X, y: X.shape[0] == y.shape[0], "X and y must have equal sample count")
@icontract.require(lambda score_func: _selector_score_func_valid(score_func), "score_func must be one of the supported scoring atoms")
@icontract.require(lambda alpha: _alpha_valid(alpha), "alpha must lie in [0, 1]")
@icontract.ensure(lambda result: _selection_state_valid(result), "selection state must contain scores and a boolean support mask")
def select_fpr_fit(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    *,
    score_func: SupportedScoreFunc = "f_classif",
    alpha: float = 0.05,
) -> UnivariateSelectionState:
    """Fit a selector that keeps features whose p-values are below alpha."""
    checked_x, checked_y = check_X_y(X, y, accept_sparse=False, dtype=np.float64)
    scores, pvalues = _selector_scores(checked_x, checked_y, score_func)
    support_mask = _support_fpr(pvalues, alpha)
    return _build_selection_state(
        checked_x,
        scores,
        pvalues,
        score_func=score_func,
        selector="fpr",
        selector_param=float(alpha),
        support_mask=support_mask,
    )


@register_atom(witness_select_fdr_fit)
@icontract.require(lambda X: X.ndim == 2, "X must be 2D")
@icontract.require(lambda X, y: X.shape[0] == y.shape[0], "X and y must have equal sample count")
@icontract.require(lambda score_func: _selector_score_func_valid(score_func), "score_func must be one of the supported scoring atoms")
@icontract.require(lambda alpha: _alpha_valid(alpha), "alpha must lie in [0, 1]")
@icontract.ensure(lambda result: _selection_state_valid(result), "selection state must contain scores and a boolean support mask")
def select_fdr_fit(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    *,
    score_func: SupportedScoreFunc = "f_classif",
    alpha: float = 0.05,
) -> UnivariateSelectionState:
    """Fit a Benjamini-Hochberg false-discovery-rate selector."""
    checked_x, checked_y = check_X_y(X, y, accept_sparse=False, dtype=np.float64)
    scores, pvalues = _selector_scores(checked_x, checked_y, score_func)
    support_mask = _support_fdr(pvalues, alpha)
    return _build_selection_state(
        checked_x,
        scores,
        pvalues,
        score_func=score_func,
        selector="fdr",
        selector_param=float(alpha),
        support_mask=support_mask,
    )


@register_atom(witness_select_fwe_fit)
@icontract.require(lambda X: X.ndim == 2, "X must be 2D")
@icontract.require(lambda X, y: X.shape[0] == y.shape[0], "X and y must have equal sample count")
@icontract.require(lambda score_func: _selector_score_func_valid(score_func), "score_func must be one of the supported scoring atoms")
@icontract.require(lambda alpha: _alpha_valid(alpha), "alpha must lie in [0, 1]")
@icontract.ensure(lambda result: _selection_state_valid(result), "selection state must contain scores and a boolean support mask")
def select_fwe_fit(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    *,
    score_func: SupportedScoreFunc = "f_classif",
    alpha: float = 0.05,
) -> UnivariateSelectionState:
    """Fit a family-wise-error selector using Bonferroni correction."""
    checked_x, checked_y = check_X_y(X, y, accept_sparse=False, dtype=np.float64)
    scores, pvalues = _selector_scores(checked_x, checked_y, score_func)
    support_mask = _support_fwe(pvalues, alpha)
    return _build_selection_state(
        checked_x,
        scores,
        pvalues,
        score_func=score_func,
        selector="fwe",
        selector_param=float(alpha),
        support_mask=support_mask,
    )


@register_atom(witness_generic_univariate_select_fit)
@icontract.require(lambda X: X.ndim == 2, "X must be 2D")
@icontract.require(lambda X, y: X.shape[0] == y.shape[0], "X and y must have equal sample count")
@icontract.require(lambda score_func: _selector_score_func_valid(score_func), "score_func must be one of the supported scoring atoms")
@icontract.require(lambda mode: mode in {"percentile", "k_best", "fpr", "fdr", "fwe"}, "mode must be a supported univariate selection strategy")
@icontract.require(lambda mode, param: _generic_mode_param_valid(mode, param), "param must match the selected univariate mode")
@icontract.ensure(lambda result: _selection_state_valid(result), "selection state must contain scores and a boolean support mask")
def generic_univariate_select_fit(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    *,
    score_func: SupportedScoreFunc = "f_classif",
    mode: str = "percentile",
    param: SelectorParam = 1e-5,
) -> UnivariateSelectionState:
    """Fit a univariate selector using a runtime-selected selection strategy."""
    checked_x, checked_y = check_X_y(X, y, accept_sparse=False, dtype=np.float64)
    scores, pvalues = _selector_scores(checked_x, checked_y, score_func)
    if mode == "k_best":
        support_mask = _support_k_best(scores, param)
    elif mode == "percentile":
        support_mask = _support_percentile(scores, float(param))
    elif mode == "fpr":
        support_mask = _support_fpr(pvalues, float(param))
    elif mode == "fdr":
        support_mask = _support_fdr(pvalues, float(param))
    else:
        support_mask = _support_fwe(pvalues, float(param))
    return _build_selection_state(
        checked_x,
        scores,
        pvalues,
        score_func=score_func,
        selector="generic",
        selector_param=param,
        support_mask=support_mask,
    )


@register_atom(witness_univariate_selection_transform)
@icontract.require(lambda X: X.ndim == 2, "X must be 2D")
@icontract.require(lambda X, state: _feature_count_matches(X, state), "X feature count must match fitted selector state")
@icontract.require(lambda state: _selection_state_valid(state), "state must be a fitted univariate selector")
@icontract.ensure(lambda result, X, state: _selection_transform_valid(result, X, state), "selected matrix must keep fitted support columns")
def univariate_selection_transform(
    X: NDArray[np.float64],
    state: UnivariateSelectionState,
) -> NDArray[np.float64]:
    """Retain the columns selected by a fitted univariate selector state."""
    checked_x = check_array(X, accept_sparse=False, dtype=np.float64)
    if not state.support_mask.any():
        return np.empty(0, dtype=checked_x.dtype).reshape((checked_x.shape[0], 0))
    return np.asarray(checked_x[:, state.support_mask], dtype=np.float64)
