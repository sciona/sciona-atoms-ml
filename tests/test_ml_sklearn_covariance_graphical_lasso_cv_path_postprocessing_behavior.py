from __future__ import annotations

import numpy as np
import pytest


def _sample_path_records() -> tuple[tuple[float, tuple[float, ...], object], ...]:
    return (
        (0.1, (0.72, 0.70), ("cov_low_fold0", "cov_low_fold1")),
        (0.4, (0.65, 0.66), ("cov_high_fold0", "cov_high_fold1")),
        (0.2, (0.74, 0.73), ("cov_mid_fold0", "cov_mid_fold1")),
    )


def test_graphical_lasso_cv_path_postprocessing_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.covariance.graphical_lasso_cv_path_postprocessing import (
        graphical_lasso_cv_alphas_with_baseline,
        graphical_lasso_cv_best_alpha,
        graphical_lasso_cv_path_alphas,
        graphical_lasso_cv_path_score_matrix,
        graphical_lasso_cv_scores_with_baseline,
        graphical_lasso_cv_sorted_path_records,
    )

    assert callable(graphical_lasso_cv_sorted_path_records)
    assert callable(graphical_lasso_cv_path_alphas)
    assert callable(graphical_lasso_cv_path_score_matrix)
    assert callable(graphical_lasso_cv_alphas_with_baseline)
    assert callable(graphical_lasso_cv_scores_with_baseline)
    assert callable(graphical_lasso_cv_best_alpha)


def test_graphical_lasso_cv_sorted_path_records_match_source_logic() -> None:
    from sciona.atoms.ml.sklearn.covariance.graphical_lasso_cv_path_postprocessing import (
        graphical_lasso_cv_sorted_path_records,
    )

    sorted_records = graphical_lasso_cv_sorted_path_records(_sample_path_records())
    assert [record[0] for record in sorted_records] == [0.4, 0.2, 0.1]
    assert sorted_records[0][2] == ("cov_high_fold0", "cov_high_fold1")


def test_graphical_lasso_cv_path_unpack_helpers_match_source_logic() -> None:
    from sciona.atoms.ml.sklearn.covariance.graphical_lasso_cv_path_postprocessing import (
        graphical_lasso_cv_path_alphas,
        graphical_lasso_cv_path_score_matrix,
        graphical_lasso_cv_sorted_path_records,
    )

    sorted_records = graphical_lasso_cv_sorted_path_records(_sample_path_records())
    alphas = graphical_lasso_cv_path_alphas(sorted_records)
    scores = graphical_lasso_cv_path_score_matrix(sorted_records)

    assert np.array_equal(alphas, np.array([0.4, 0.2, 0.1], dtype=np.float64))
    assert np.allclose(
        scores,
        np.array([[0.65, 0.66], [0.74, 0.73], [0.72, 0.70]], dtype=np.float64),
    )


def test_graphical_lasso_cv_baseline_append_helpers_match_source_logic() -> None:
    from sciona.atoms.ml.sklearn.covariance.graphical_lasso_cv_path_postprocessing import (
        graphical_lasso_cv_alphas_with_baseline,
        graphical_lasso_cv_scores_with_baseline,
    )

    alphas = np.array([0.4, 0.2, 0.1], dtype=np.float64)
    grid_scores = np.array([[0.65, 0.66], [0.74, 0.73], [0.72, 0.70]], dtype=np.float64)
    empirical_scores = np.array([0.58, 0.60], dtype=np.float64)

    assert np.array_equal(
        graphical_lasso_cv_alphas_with_baseline(alphas),
        np.array([0.4, 0.2, 0.1, 0.0], dtype=np.float64),
    )
    assert np.allclose(
        graphical_lasso_cv_scores_with_baseline(grid_scores, empirical_scores),
        np.array(
            [[0.65, 0.66], [0.74, 0.73], [0.72, 0.70], [0.58, 0.60]],
            dtype=np.float64,
        ),
    )


def test_graphical_lasso_cv_best_alpha_matches_source_logic() -> None:
    from sciona.atoms.ml.sklearn.covariance.graphical_lasso_cv_path_postprocessing import (
        graphical_lasso_cv_alphas_with_baseline,
        graphical_lasso_cv_best_alpha,
    )

    alphas = graphical_lasso_cv_alphas_with_baseline(np.array([0.4, 0.2, 0.1], dtype=np.float64))
    assert graphical_lasso_cv_best_alpha(alphas, 1) == pytest.approx(0.2)


def test_graphical_lasso_cv_path_postprocessing_contracts_reject_invalid_inputs() -> None:
    from sciona.atoms.ml.sklearn.covariance.graphical_lasso_cv_path_postprocessing import (
        graphical_lasso_cv_alphas_with_baseline,
        graphical_lasso_cv_best_alpha,
        graphical_lasso_cv_path_score_matrix,
        graphical_lasso_cv_scores_with_baseline,
        graphical_lasso_cv_sorted_path_records,
    )

    with pytest.raises(Exception):
        graphical_lasso_cv_sorted_path_records([])
    with pytest.raises(Exception):
        graphical_lasso_cv_path_score_matrix(((0.4, (0.1, 0.2), None), (0.2, (0.3,), None)))
    with pytest.raises(Exception):
        graphical_lasso_cv_alphas_with_baseline(np.array([0.2, 0.4], dtype=np.float64))
    with pytest.raises(Exception):
        graphical_lasso_cv_scores_with_baseline(np.array([[0.1, 0.2]], dtype=np.float64), np.array([0.3], dtype=np.float64))
    with pytest.raises(Exception):
        graphical_lasso_cv_best_alpha(np.array([0.4, 0.2, 0.0], dtype=np.float64), 4)
