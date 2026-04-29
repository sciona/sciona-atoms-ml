from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError
from sklearn.covariance._graph_lasso import alpha_max

from sciona.atoms.ml.sklearn.covariance.graphical_lasso_cv_bookkeeping import (
    graphical_lasso_cv_alpha_grid,
    graphical_lasso_cv_best_index,
    graphical_lasso_cv_mean_test_scores,
    graphical_lasso_cv_refined_alpha_grid,
    graphical_lasso_cv_refinement_bounds,
    graphical_lasso_cv_results,
)


def _emp_cov() -> np.ndarray:
    return np.array(
        [
            [2.0, 0.3, 0.1],
            [0.3, 1.5, 0.2],
            [0.1, 0.2, 1.0],
        ],
        dtype=np.float64,
    )


def test_graphical_lasso_cv_alpha_grid_matches_sklearn_formula() -> None:
    emp_cov = _emp_cov()
    observed = graphical_lasso_cv_alpha_grid(emp_cov, n_alphas=5)
    alpha_1 = alpha_max(emp_cov)
    alpha_0 = 1e-2 * alpha_1
    expected = np.logspace(np.log10(alpha_0), np.log10(alpha_1), 5)[::-1]
    assert np.allclose(observed, expected)


def test_graphical_lasso_cv_mean_scores_and_best_index_follow_sklearn_tie_rule() -> None:
    grid_scores = np.array(
        [
            [1.0, 1.0],
            [2.0, 2.0],
            [2.0, 2.0],
        ],
        dtype=np.float64,
    )
    means = graphical_lasso_cv_mean_test_scores(grid_scores)
    assert np.allclose(means, np.array([1.0, 2.0, 2.0], dtype=np.float64))
    assert graphical_lasso_cv_best_index(means) == 2


def test_graphical_lasso_cv_mean_scores_clip_overflow_to_nan() -> None:
    huge = 0.1 / np.finfo(np.float64).eps
    grid_scores = np.array([[huge, huge], [1.0, 2.0]], dtype=np.float64)
    means = graphical_lasso_cv_mean_test_scores(grid_scores)
    assert np.isnan(means[0])
    assert np.isclose(means[1], 1.5)


@pytest.mark.parametrize(
    ("alphas", "mean_scores", "expected"),
    [
        (np.array([10.0, 5.0, 2.0], dtype=np.float64), np.array([3.0, 2.0, 1.0], dtype=np.float64), (10.0, 5.0)),
        (np.array([10.0, 5.0, 2.0, 1.0], dtype=np.float64), np.array([np.nan, 4.0, np.nan, np.nan], dtype=np.float64), (5.0, 2.0)),
        (np.array([10.0, 5.0, 2.0], dtype=np.float64), np.array([1.0, 2.0, 3.0], dtype=np.float64), (2.0, 0.02)),
        (np.array([10.0, 5.0, 2.0, 1.0], dtype=np.float64), np.array([1.0, 4.0, 3.0, 2.0], dtype=np.float64), (10.0, 2.0)),
    ],
)
def test_graphical_lasso_cv_refinement_bounds_cover_all_fit_cases(
    alphas: np.ndarray,
    mean_scores: np.ndarray,
    expected: tuple[float, float],
) -> None:
    observed = graphical_lasso_cv_refinement_bounds(alphas, mean_scores)
    assert observed == pytest.approx(expected)


def test_graphical_lasso_cv_refined_alpha_grid_matches_sklearn_formula() -> None:
    observed = graphical_lasso_cv_refined_alpha_grid(10.0, 2.0, n_alphas=4)
    expected = np.logspace(np.log10(10.0), np.log10(2.0), 6)[1:-1]
    assert np.allclose(observed, expected)


def test_graphical_lasso_cv_results_materializes_split_and_summary_scores() -> None:
    alphas = np.array([3.0, 1.0, 0.0], dtype=np.float64)
    grid_scores = np.array(
        [
            [0.2, 0.4],
            [0.5, 0.1],
            [0.0, 0.3],
        ],
        dtype=np.float64,
    )
    results = graphical_lasso_cv_results(alphas, grid_scores)
    assert np.array_equal(results["alphas"], alphas)
    assert np.array_equal(results["split0_test_score"], grid_scores[:, 0])
    assert np.array_equal(results["split1_test_score"], grid_scores[:, 1])
    assert np.allclose(results["mean_test_score"], np.mean(grid_scores, axis=1))
    assert np.allclose(results["std_test_score"], np.std(grid_scores, axis=1))


def test_graphical_lasso_cv_bookkeeping_rejects_invalid_inputs() -> None:
    emp_cov = _emp_cov()
    with pytest.raises((ViolationError, ValueError)):
        graphical_lasso_cv_alpha_grid(emp_cov, n_alphas=1)

    with pytest.raises((ViolationError, ValueError)):
        graphical_lasso_cv_best_index(np.array([np.nan, np.nan], dtype=np.float64))

    with pytest.raises((ViolationError, ValueError)):
        graphical_lasso_cv_refinement_bounds(
            np.array([5.0, 4.0], dtype=np.float64),
            np.array([1.0, 2.0, 3.0], dtype=np.float64),
        )

    with pytest.raises((ViolationError, ValueError)):
        graphical_lasso_cv_refined_alpha_grid(1.0, 2.0, n_alphas=4)

    with pytest.raises((ViolationError, ValueError)):
        graphical_lasso_cv_results(
            np.array([2.0, 1.0], dtype=np.float64),
            np.array([[0.1], [0.2], [0.3]], dtype=np.float64),
        )
