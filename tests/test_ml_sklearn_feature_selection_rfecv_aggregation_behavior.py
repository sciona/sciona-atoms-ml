from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError


def test_rfecv_aggregation_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.feature_selection.rfecv_aggregation import (
        rfecv_best_feature_count,
        rfecv_cv_results,
    )

    assert callable(rfecv_best_feature_count)
    assert callable(rfecv_cv_results)


def test_rfecv_best_feature_count_matches_reverse_tie_break_logic() -> None:
    from sciona.atoms.ml.sklearn.feature_selection.rfecv_aggregation import rfecv_best_feature_count

    step_scores = np.array(
        [
            [0.70, 0.80, 0.90],
            [0.80, 0.80, 0.70],
        ],
        dtype=np.float64,
    )
    step_n_features = np.array(
        [
            [5, 3, 1],
            [5, 3, 1],
        ],
        dtype=np.int64,
    )

    expected = int(np.asarray(step_n_features[0])[::-1][np.argmax(np.sum(step_scores, axis=0)[::-1])])
    assert rfecv_best_feature_count(step_scores, step_n_features) == expected
    assert expected == 1


def test_rfecv_cv_results_matches_source_materialization_and_1d_path() -> None:
    from sciona.atoms.ml.sklearn.feature_selection.rfecv_aggregation import rfecv_cv_results

    step_scores = np.array(
        [
            [0.42, 0.51, 0.61, 0.58],
            [0.40, 0.55, 0.60, 0.62],
            [0.44, 0.52, 0.59, 0.63],
        ],
        dtype=np.float64,
    )
    step_n_features = np.array([7, 5, 3, 1], dtype=np.int64)

    scores_rev = step_scores[:, ::-1]
    expected = {
        "mean_test_score": np.mean(scores_rev, axis=0),
        "std_test_score": np.std(scores_rev, axis=0),
        **{f"split{i}_test_score": scores_rev[i] for i in range(step_scores.shape[0])},
        "n_features": step_n_features[::-1],
    }

    result = rfecv_cv_results(step_scores, step_n_features)
    assert list(result) == list(expected)
    for key, expected_value in expected.items():
        assert np.allclose(result[key], expected_value)


def test_contracts_reject_inconsistent_or_invalid_step_paths() -> None:
    from sciona.atoms.ml.sklearn.feature_selection.rfecv_aggregation import (
        rfecv_best_feature_count,
        rfecv_cv_results,
    )

    step_scores = np.array([[0.6, 0.7, 0.8], [0.5, 0.7, 0.9]], dtype=np.float64)

    with pytest.raises(ViolationError):
        rfecv_best_feature_count(step_scores, np.array([[6, 4, 2], [6, 3, 2]], dtype=np.int64))

    with pytest.raises(ViolationError):
        rfecv_cv_results(step_scores, np.array([2, 4, 6], dtype=np.int64))
