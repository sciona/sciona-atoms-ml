from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier


def test_forest_feature_importances_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.ensemble.forest_feature_importances import (
        forest_average_feature_importances,
        forest_importance_contributor_mask,
        forest_normalized_feature_importances,
        forest_zero_feature_importances,
    )

    assert callable(forest_importance_contributor_mask)
    assert callable(forest_zero_feature_importances)
    assert callable(forest_average_feature_importances)
    assert callable(forest_normalized_feature_importances)


def test_forest_feature_importances_match_sklearn_property() -> None:
    from sciona.atoms.ml.sklearn.ensemble.forest_feature_importances import (
        forest_average_feature_importances,
        forest_importance_contributor_mask,
        forest_normalized_feature_importances,
    )

    X, y = make_classification(
        n_samples=40,
        n_features=5,
        n_informative=4,
        n_redundant=0,
        random_state=0,
    )
    clf = RandomForestClassifier(n_estimators=5, random_state=0, max_depth=2).fit(X, y)

    node_counts = np.asarray([tree.tree_.node_count for tree in clf.estimators_], dtype=np.int64)
    mask = forest_importance_contributor_mask(node_counts)
    blocks = tuple(
        np.asarray(tree.feature_importances_, dtype=np.float64)
        for tree, include in zip(clf.estimators_, mask)
        if include
    )
    averaged = forest_average_feature_importances(blocks)
    normalized = forest_normalized_feature_importances(averaged)

    assert np.array_equal(mask, node_counts > 1)
    assert np.allclose(normalized, clf.feature_importances_)


def test_forest_zero_feature_importances_matches_all_root_case() -> None:
    from sciona.atoms.ml.sklearn.ensemble.forest_feature_importances import (
        forest_importance_contributor_mask,
        forest_zero_feature_importances,
    )

    X = np.arange(30, dtype=np.float64).reshape(15, 2)
    y = np.zeros(15, dtype=np.int64)
    clf = RandomForestClassifier(n_estimators=4, random_state=0).fit(X, y)

    node_counts = np.asarray([tree.tree_.node_count for tree in clf.estimators_], dtype=np.int64)
    mask = forest_importance_contributor_mask(node_counts)
    zero = forest_zero_feature_importances(int(clf.n_features_in_))

    assert np.array_equal(mask, np.zeros_like(node_counts, dtype=np.bool_))
    assert np.allclose(zero, clf.feature_importances_)


def test_forest_average_and_normalized_feature_importances_shapes() -> None:
    from sciona.atoms.ml.sklearn.ensemble.forest_feature_importances import (
        forest_average_feature_importances,
        forest_normalized_feature_importances,
    )

    blocks = (
        np.array([0.2, 0.3, 0.5], dtype=np.float64),
        np.array([0.1, 0.4, 0.5], dtype=np.float64),
    )
    averaged = forest_average_feature_importances(blocks)
    normalized = forest_normalized_feature_importances(averaged)

    assert np.allclose(averaged, np.array([0.15, 0.35, 0.5], dtype=np.float64))
    assert np.allclose(normalized, averaged)


def test_contracts_reject_invalid_forest_feature_importance_inputs() -> None:
    from sciona.atoms.ml.sklearn.ensemble.forest_feature_importances import (
        forest_average_feature_importances,
        forest_importance_contributor_mask,
        forest_normalized_feature_importances,
        forest_zero_feature_importances,
    )

    with pytest.raises(ViolationError):
        forest_importance_contributor_mask(np.array([0, 2], dtype=np.int64))

    with pytest.raises(ViolationError):
        forest_zero_feature_importances(0)

    with pytest.raises(ViolationError):
        forest_average_feature_importances(
            (
                np.array([0.2, 0.8], dtype=np.float64),
                np.array([0.2, 0.3, 0.5], dtype=np.float64),
            )
        )

    with pytest.raises(ViolationError):
        forest_normalized_feature_importances(np.array([0.0, 0.0], dtype=np.float64))
