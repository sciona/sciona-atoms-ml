"""GraphicalLassoCV bookkeeping helper atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray
from sklearn.covariance._graph_lasso import alpha_max

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_graphical_lasso_cv_alpha_grid,
    witness_graphical_lasso_cv_best_index,
    witness_graphical_lasso_cv_mean_test_scores,
    witness_graphical_lasso_cv_refined_alpha_grid,
    witness_graphical_lasso_cv_refinement_bounds,
    witness_graphical_lasso_cv_results,
)


def _finite_square_matrix(values: object) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(
        array.ndim == 2
        and array.shape[0] >= 2
        and array.shape[0] == array.shape[1]
        and np.all(np.isfinite(array))
    )


def _positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


def _alpha_count_valid(value: object) -> bool:
    return _positive_int(value) and int(value) >= 2


def _finite_vector(values: object) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 1 and array.shape[0] >= 1 and np.all(np.isfinite(array)))


def _strictly_positive_vector(values: object) -> bool:
    return bool(_finite_vector(values) and np.all(np.asarray(values, dtype=np.float64) > 0.0))


def _descending_unique_vector(values: object) -> bool:
    if not _strictly_positive_vector(values):
        return False
    array = np.asarray(values, dtype=np.float64)
    return bool(np.all(array[:-1] > array[1:]))


def _descending_nonnegative_unique_vector(values: object) -> bool:
    if not _finite_vector(values):
        return False
    array = np.asarray(values, dtype=np.float64)
    return bool(np.all(array >= 0.0) and np.all(array[:-1] > array[1:]))


def _score_matrix_valid(values: object) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 2 and array.shape[0] >= 1 and array.shape[1] >= 1 and np.all(np.isfinite(array)))


def _score_rows_match_alphas(alphas: object, grid_scores: object) -> bool:
    return bool(
        _descending_nonnegative_unique_vector(alphas)
        and _score_matrix_valid(grid_scores)
        and np.asarray(alphas, dtype=np.float64).shape[0]
        == np.asarray(grid_scores, dtype=np.float64).shape[0]
    )


def _finite_or_nan_vector(values: object) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 1 and array.shape[0] >= 1 and np.all(np.isnan(array) | np.isfinite(array)))


def _mean_scores_match(result: object, grid_scores: object) -> bool:
    values = np.asarray(result, dtype=np.float64)
    scores = np.asarray(grid_scores, dtype=np.float64)
    return bool(values.shape == (scores.shape[0],) and _finite_or_nan_vector(result))


def _index_valid(result: object, mean_scores: object) -> bool:
    return isinstance(result, int) and 0 <= result < np.asarray(mean_scores, dtype=np.float64).shape[0]


def _bounds_valid(result: object, alphas: object) -> bool:
    if not (isinstance(result, tuple) and len(result) == 2):
        return False
    upper, lower = result
    return bool(np.isfinite(float(upper)) and np.isfinite(float(lower)) and float(upper) > float(lower) > 0.0 and float(upper) <= np.asarray(alphas, dtype=np.float64)[0])


def _results_valid(result: object, alphas: object, grid_scores: object) -> bool:
    if not isinstance(result, dict):
        return False
    alpha_values = np.asarray(alphas, dtype=np.float64)
    score_values = np.asarray(grid_scores, dtype=np.float64)
    required = {"alphas", "mean_test_score", "std_test_score"}
    split_keys = {f"split{i}_test_score" for i in range(score_values.shape[1])}
    if set(result) != required | split_keys:
        return False
    if not np.array_equal(np.asarray(result["alphas"], dtype=np.float64), alpha_values):
        return False
    if not np.allclose(np.asarray(result["mean_test_score"], dtype=np.float64), np.mean(score_values, axis=1)):
        return False
    if not np.allclose(np.asarray(result["std_test_score"], dtype=np.float64), np.std(score_values, axis=1)):
        return False
    return all(
        np.allclose(np.asarray(result[f"split{i}_test_score"], dtype=np.float64), score_values[:, i])
        for i in range(score_values.shape[1])
    )


@register_atom(witness_graphical_lasso_cv_alpha_grid)
@icontract.require(lambda emp_cov: _finite_square_matrix(emp_cov), "emp_cov must be a finite square covariance matrix with at least two features")
@icontract.require(lambda n_alphas: _alpha_count_valid(n_alphas), "n_alphas must be an integer >= 2")
@icontract.require(lambda alpha_min_ratio: isinstance(alpha_min_ratio, (int, float)) and not isinstance(alpha_min_ratio, bool) and np.isfinite(float(alpha_min_ratio)) and 0.0 < float(alpha_min_ratio) < 1.0, "alpha_min_ratio must lie strictly between zero and one")
@icontract.ensure(lambda result: _descending_unique_vector(result), "alpha grid must be a strictly decreasing positive vector")
def graphical_lasso_cv_alpha_grid(
    emp_cov: NDArray[np.float64],
    *,
    n_alphas: int,
    alpha_min_ratio: float = 1e-2,
) -> NDArray[np.float64]:
    """Build GraphicalLassoCV's initial descending logarithmic alpha grid."""
    emp_cov_values = np.asarray(emp_cov, dtype=np.float64)
    alpha_upper = float(alpha_max(emp_cov_values))
    alpha_lower = float(alpha_min_ratio) * alpha_upper
    return np.asarray(np.logspace(np.log10(alpha_lower), np.log10(alpha_upper), int(n_alphas))[::-1], dtype=np.float64)


@register_atom(witness_graphical_lasso_cv_mean_test_scores)
@icontract.require(lambda grid_scores: _score_matrix_valid(grid_scores), "grid_scores must be a finite alpha-by-fold score matrix")
@icontract.require(lambda score_overflow_threshold: isinstance(score_overflow_threshold, (int, float)) and not isinstance(score_overflow_threshold, bool) and np.isfinite(float(score_overflow_threshold)), "score_overflow_threshold must be finite")
@icontract.ensure(lambda result, grid_scores: _mean_scores_match(result, grid_scores), "mean test scores must produce one finite-or-NaN value per alpha")
def graphical_lasso_cv_mean_test_scores(
    grid_scores: NDArray[np.float64],
    *,
    score_overflow_threshold: float = 0.1 / np.finfo(np.float64).eps,
) -> NDArray[np.float64]:
    """Compute GraphicalLassoCV's per-alpha mean scores with overflow-to-NaN clipping."""
    scores = np.asarray(grid_scores, dtype=np.float64)
    means = np.mean(scores, axis=1)
    means = np.asarray(means, dtype=np.float64)
    means[means >= float(score_overflow_threshold)] = np.nan
    return means


@register_atom(witness_graphical_lasso_cv_best_index)
@icontract.require(lambda mean_test_scores: _finite_or_nan_vector(mean_test_scores), "mean_test_scores must be a 1D finite-or-NaN score vector")
@icontract.require(lambda mean_test_scores: np.any(np.isfinite(np.asarray(mean_test_scores, dtype=np.float64))), "mean_test_scores must contain at least one finite value")
@icontract.ensure(lambda result, mean_test_scores: _index_valid(result, mean_test_scores), "best index must point into mean_test_scores")
def graphical_lasso_cv_best_index(
    mean_test_scores: NDArray[np.float64],
) -> int:
    """Select GraphicalLassoCV's best alpha index with later-equal-score tie breaking."""
    means = np.asarray(mean_test_scores, dtype=np.float64)
    best_score = -np.inf
    best_index = 0
    for index, score in enumerate(means):
        if np.isfinite(score) and score >= best_score:
            best_score = float(score)
            best_index = index
    return int(best_index)


@register_atom(witness_graphical_lasso_cv_refinement_bounds)
@icontract.require(lambda alphas, mean_test_scores: _descending_unique_vector(alphas), "alphas must be a strictly decreasing positive vector")
@icontract.require(lambda alphas, mean_test_scores: _finite_or_nan_vector(mean_test_scores) and np.asarray(mean_test_scores, dtype=np.float64).shape == np.asarray(alphas, dtype=np.float64).shape, "mean_test_scores must align one-to-one with alphas")
@icontract.require(lambda mean_test_scores: np.any(np.isfinite(np.asarray(mean_test_scores, dtype=np.float64))), "mean_test_scores must contain at least one finite value")
@icontract.ensure(lambda result, alphas: _bounds_valid(result, alphas), "refinement bounds must be a descending positive alpha interval")
def graphical_lasso_cv_refinement_bounds(
    alphas: NDArray[np.float64],
    mean_test_scores: NDArray[np.float64],
) -> tuple[float, float]:
    """Select GraphicalLassoCV's next alpha interval from the current scored path."""
    alpha_values = np.asarray(alphas, dtype=np.float64)
    mean_values = np.asarray(mean_test_scores, dtype=np.float64)
    best_index = graphical_lasso_cv_best_index(mean_values)
    finite_indices = np.flatnonzero(np.isfinite(mean_values))
    last_finite_idx = int(finite_indices[-1])

    if best_index == 0:
        alpha_upper = alpha_values[0]
        alpha_lower = alpha_values[1]
    elif best_index == last_finite_idx and best_index != len(alpha_values) - 1:
        alpha_upper = alpha_values[best_index]
        alpha_lower = alpha_values[best_index + 1]
    elif best_index == len(alpha_values) - 1:
        alpha_upper = alpha_values[best_index]
        alpha_lower = 0.01 * alpha_values[best_index]
    else:
        alpha_upper = alpha_values[best_index - 1]
        alpha_lower = alpha_values[best_index + 1]

    return float(alpha_upper), float(alpha_lower)


@register_atom(witness_graphical_lasso_cv_refined_alpha_grid)
@icontract.require(lambda alpha_upper: isinstance(alpha_upper, (int, float)) and not isinstance(alpha_upper, bool) and np.isfinite(float(alpha_upper)) and float(alpha_upper) > 0.0, "alpha_upper must be a positive finite scalar")
@icontract.require(lambda alpha_lower: isinstance(alpha_lower, (int, float)) and not isinstance(alpha_lower, bool) and np.isfinite(float(alpha_lower)) and float(alpha_lower) > 0.0, "alpha_lower must be a positive finite scalar")
@icontract.require(lambda alpha_upper, alpha_lower: float(alpha_upper) > float(alpha_lower), "alpha_upper must be greater than alpha_lower")
@icontract.require(lambda n_alphas: _alpha_count_valid(n_alphas), "n_alphas must be an integer >= 2")
@icontract.ensure(lambda result: _descending_unique_vector(result), "refined alpha grid must be a strictly decreasing positive vector")
def graphical_lasso_cv_refined_alpha_grid(
    alpha_upper: float,
    alpha_lower: float,
    *,
    n_alphas: int,
) -> NDArray[np.float64]:
    """Build GraphicalLassoCV's interior refinement grid between two alpha bounds."""
    values = np.logspace(np.log10(float(alpha_upper)), np.log10(float(alpha_lower)), int(n_alphas) + 2)
    return np.asarray(values[1:-1], dtype=np.float64)


@register_atom(witness_graphical_lasso_cv_results)
@icontract.require(lambda alphas, grid_scores: _score_rows_match_alphas(alphas, grid_scores), "alphas and grid_scores must align by alpha rows")
@icontract.ensure(lambda result, alphas, grid_scores: _results_valid(result, alphas, grid_scores), "cv_results must expose alphas, per-split scores, and score aggregates")
def graphical_lasso_cv_results(
    alphas: NDArray[np.float64],
    grid_scores: NDArray[np.float64],
) -> dict[str, NDArray[np.float64]]:
    """Materialize GraphicalLassoCV's `cv_results_` fields from alpha-by-fold scores."""
    alpha_values = np.asarray(alphas, dtype=np.float64)
    score_values = np.asarray(grid_scores, dtype=np.float64)
    results: dict[str, NDArray[np.float64]] = {"alphas": alpha_values.copy()}
    for i in range(score_values.shape[1]):
        results[f"split{i}_test_score"] = np.asarray(score_values[:, i], dtype=np.float64)
    results["mean_test_score"] = np.asarray(np.mean(score_values, axis=1), dtype=np.float64)
    results["std_test_score"] = np.asarray(np.std(score_values, axis=1), dtype=np.float64)
    return results
