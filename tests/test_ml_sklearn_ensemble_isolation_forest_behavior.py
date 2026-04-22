from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError
from sklearn.ensemble import IsolationForest
from sklearn.ensemble._iforest import _average_path_length


def test_isolation_forest_helper_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.ensemble.isolation_forest import (
        isolation_forest_average_path_length,
        isolation_forest_leaf_depths,
        isolation_forest_raw_scores,
    )

    assert callable(isolation_forest_average_path_length)
    assert callable(isolation_forest_leaf_depths)
    assert callable(isolation_forest_raw_scores)


def test_average_path_length_matches_sklearn_private_helper() -> None:
    from sciona.atoms.ml.sklearn.ensemble.isolation_forest import isolation_forest_average_path_length

    counts = np.array([0, 1, 2, 3, 10, 64, 256], dtype=np.int64)
    result = isolation_forest_average_path_length(counts)

    assert np.allclose(result, _average_path_length(counts))
    assert result[0] == 0.0
    assert result[1] == 0.0
    assert result[2] == 1.0


def test_leaf_depths_match_sklearn_depth_formula() -> None:
    from sciona.atoms.ml.sklearn.ensemble.isolation_forest import (
        isolation_forest_average_path_length,
        isolation_forest_leaf_depths,
    )

    leaf_indices = np.array([0, 2, 3, 1], dtype=np.int64)
    decision_path_lengths = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    node_sample_counts = np.array([8, 4, 2, 1], dtype=np.int64)
    average_path_lengths = isolation_forest_average_path_length(node_sample_counts)

    result = isolation_forest_leaf_depths(leaf_indices, decision_path_lengths, average_path_lengths)
    expected = decision_path_lengths[leaf_indices] + average_path_lengths[leaf_indices] - 1.0
    assert np.allclose(result, expected)


def test_raw_scores_match_sklearn_private_score_formula_on_fitted_forest() -> None:
    from sciona.atoms.ml.sklearn.ensemble.isolation_forest import (
        isolation_forest_leaf_depths,
        isolation_forest_raw_scores,
    )

    X = np.array(
        [
            [-1.1, 0.1],
            [0.2, 0.0],
            [0.3, -0.2],
            [0.4, 0.2],
            [100.0, 100.0],
            [0.5, 0.3],
        ],
        dtype=np.float32,
    )
    forest = IsolationForest(n_estimators=7, max_samples=5, random_state=13).fit(X)
    depths = np.zeros(X.shape[0], order="f")
    for tree, features, decision_lengths, average_lengths in zip(
        forest.estimators_,
        forest.estimators_features_,
        forest._decision_path_lengths,
        forest._average_path_length_per_tree,
    ):
        leaves = tree.apply(X[:, features], check_input=False)
        depths += isolation_forest_leaf_depths(
            leaves.astype(np.int64),
            np.asarray(decision_lengths, dtype=np.float64),
            np.asarray(average_lengths, dtype=np.float64),
        )

    result = isolation_forest_raw_scores(depths, n_estimators=len(forest.estimators_), max_samples=forest._max_samples)
    expected = forest._compute_score_samples(X, subsample_features=False)
    assert np.allclose(result, expected)
    assert np.allclose(forest.score_samples(X), -result)


def test_raw_scores_match_sklearn_zero_denominator_case() -> None:
    from sciona.atoms.ml.sklearn.ensemble.isolation_forest import isolation_forest_raw_scores

    depths = np.array([0.0, 1.0], dtype=np.float64)
    result = isolation_forest_raw_scores(depths, n_estimators=3, max_samples=1)
    denominator = 3 * _average_path_length([1])
    expected = 2 ** (-np.divide(depths, denominator, out=np.ones_like(depths), where=denominator != 0))
    assert np.allclose(result, expected)


def test_contracts_reject_invalid_isolation_forest_inputs() -> None:
    from sciona.atoms.ml.sklearn.ensemble.isolation_forest import (
        isolation_forest_average_path_length,
        isolation_forest_leaf_depths,
        isolation_forest_raw_scores,
    )

    with pytest.raises(ViolationError):
        isolation_forest_average_path_length(np.array([1.5, 2.0], dtype=np.float64))

    with pytest.raises(ViolationError):
        isolation_forest_leaf_depths(
            np.array([3], dtype=np.int64),
            np.array([1.0, 2.0], dtype=np.float64),
            np.array([0.0, 1.0], dtype=np.float64),
        )

    with pytest.raises(ViolationError):
        isolation_forest_raw_scores(np.array([0.0, -1.0], dtype=np.float64), n_estimators=2, max_samples=8)
