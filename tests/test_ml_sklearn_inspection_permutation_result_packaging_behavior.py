from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError
from sklearn.inspection import permutation_importance as sklearn_permutation_importance
from sklearn.inspection._permutation_importance import (
    _calculate_permutation_scores,
    _weights_scorer,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import check_scoring
from sklearn.utils import Bunch, check_random_state


def _data() -> tuple[np.ndarray, np.ndarray]:
    X = np.array(
        [
            [1.0, 9.0, 9.0],
            [1.0, 9.0, 8.0],
            [1.0, 8.0, 9.0],
            [0.0, 9.0, 9.0],
            [0.0, 8.0, 9.0],
            [0.0, 9.0, 8.0],
        ],
        dtype=np.float64,
    )
    y = np.array([1, 1, 1, 0, 0, 0], dtype=np.int64)
    return X, y


def test_permutation_result_packaging_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.inspection.permutation_result_packaging import (
        permutation_importance_metric_score_matrix,
        permutation_importance_multi_metric_bunches,
        permutation_importance_random_seed,
        permutation_importance_summary_bunch,
    )

    assert callable(permutation_importance_random_seed)
    assert callable(permutation_importance_metric_score_matrix)
    assert callable(permutation_importance_summary_bunch)
    assert callable(permutation_importance_multi_metric_bunches)


def test_permutation_result_packaging_matches_sklearn_outputs() -> None:
    from sciona.atoms.ml.sklearn.inspection.permutation_result_packaging import (
        permutation_importance_metric_score_matrix,
        permutation_importance_multi_metric_bunches,
        permutation_importance_random_seed,
        permutation_importance_summary_bunch,
    )

    X, y = _data()
    estimator = LogisticRegression().fit(X, y)

    derived_seed = permutation_importance_random_seed(0)
    expected_seed = int(check_random_state(0).randint(np.iinfo(np.int32).max + 1))
    assert derived_seed == expected_seed

    single = sklearn_permutation_importance(
        estimator,
        X,
        y,
        n_repeats=3,
        random_state=0,
    )
    single_bunch = permutation_importance_summary_bunch(
        float(estimator.score(X, y)),
        np.asarray(single.importances, dtype=np.float64) * -1 + float(estimator.score(X, y)),
    )
    assert isinstance(single_bunch, Bunch)
    assert np.allclose(single_bunch["importances"], single.importances)
    assert np.allclose(single_bunch["importances_mean"], single.importances_mean)
    assert np.allclose(single_bunch["importances_std"], single.importances_std)

    scoring = ["accuracy", "neg_log_loss"]
    scorer = check_scoring(estimator, scoring=scoring)
    baseline_scores = _weights_scorer(scorer, estimator, X, y, None)
    score_dicts_by_feature = tuple(
        _calculate_permutation_scores(
            estimator,
            X,
            y,
            None,
            col_idx,
            derived_seed,
            3,
            scorer,
            X.shape[0],
        )
        for col_idx in range(X.shape[1])
    )

    accuracy_matrix = permutation_importance_metric_score_matrix(score_dicts_by_feature, "accuracy")
    assert accuracy_matrix.shape == (X.shape[1], 3)

    multi = permutation_importance_multi_metric_bunches(baseline_scores, score_dicts_by_feature)
    expected_multi = sklearn_permutation_importance(
        estimator,
        X,
        y,
        scoring=scoring,
        n_repeats=3,
        random_state=0,
    )
    assert set(multi.keys()) == set(expected_multi.keys())
    for metric_name in expected_multi:
        assert isinstance(multi[metric_name], Bunch)
        assert np.allclose(multi[metric_name]["importances"], expected_multi[metric_name].importances)
        assert np.allclose(multi[metric_name]["importances_mean"], expected_multi[metric_name].importances_mean)
        assert np.allclose(multi[metric_name]["importances_std"], expected_multi[metric_name].importances_std)


def test_permutation_result_packaging_contracts_reject_invalid_inputs() -> None:
    from sciona.atoms.ml.sklearn.inspection.permutation_result_packaging import (
        permutation_importance_metric_score_matrix,
        permutation_importance_multi_metric_bunches,
        permutation_importance_random_seed,
        permutation_importance_summary_bunch,
    )

    with pytest.raises(ViolationError):
        permutation_importance_random_seed(True)

    with pytest.raises(ViolationError):
        permutation_importance_metric_score_matrix(tuple(), "accuracy")

    with pytest.raises(ViolationError):
        permutation_importance_summary_bunch(1.0, np.array([1.0, 2.0], dtype=np.float64))

    with pytest.raises(ViolationError):
        permutation_importance_multi_metric_bunches({"accuracy": 1.0}, ({"neg_log_loss": np.array([0.1])},))
