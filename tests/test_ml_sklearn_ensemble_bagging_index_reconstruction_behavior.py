from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier


def test_bagging_index_reconstruction_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.ensemble.bagging_index_reconstruction import (
        bagging_estimator_index_pairs,
        bagging_estimators_feature_indices,
        bagging_estimators_sample_indices,
    )

    assert callable(bagging_estimator_index_pairs)
    assert callable(bagging_estimators_feature_indices)
    assert callable(bagging_estimators_sample_indices)


def test_bagging_estimator_index_pairs_match_sklearn() -> None:
    from sciona.atoms.ml.sklearn.ensemble.bagging_index_reconstruction import (
        bagging_estimator_index_pairs,
    )

    X = np.arange(60.0, dtype=np.float64).reshape(15, 4)
    y = np.tile(np.array([0, 1, 0], dtype=np.int64), 5)
    clf = BaggingClassifier(
        estimator=DecisionTreeClassifier(max_depth=1, random_state=0),
        n_estimators=4,
        bootstrap=True,
        bootstrap_features=True,
        max_samples=6,
        max_features=3,
        random_state=9,
    ).fit(X, y)

    observed = bagging_estimator_index_pairs(
        tuple(int(seed) for seed in clf._seeds),
        clf.bootstrap_features,
        clf.bootstrap,
        clf.n_features_in_,
        clf._n_samples,
        clf._max_features,
        clf._max_samples,
    )
    expected = tuple(
        (
            np.asarray(feature_indices, dtype=np.int64),
            np.asarray(sample_indices, dtype=np.int64),
        )
        for feature_indices, sample_indices in clf._get_estimators_indices()
    )

    assert len(observed) == len(expected)
    for (obs_f, obs_s), (exp_f, exp_s) in zip(observed, expected):
        assert np.array_equal(obs_f, exp_f)
        assert np.array_equal(obs_s, exp_s)


def test_bagging_estimators_feature_and_sample_indices_match_sklearn() -> None:
    from sciona.atoms.ml.sklearn.ensemble.bagging_index_reconstruction import (
        bagging_estimators_feature_indices,
        bagging_estimators_sample_indices,
    )

    X = np.arange(72.0, dtype=np.float64).reshape(12, 6)
    y = np.tile(np.array([0, 1, 1], dtype=np.int64), 4)
    clf = BaggingClassifier(
        estimator=DecisionTreeClassifier(max_depth=2, random_state=0),
        n_estimators=5,
        bootstrap=False,
        bootstrap_features=True,
        max_samples=8,
        max_features=4,
        random_state=17,
    ).fit(X, y)

    feature_blocks = bagging_estimators_feature_indices(
        tuple(int(seed) for seed in clf._seeds),
        clf.bootstrap_features,
        clf.bootstrap,
        clf.n_features_in_,
        clf._n_samples,
        clf._max_features,
        clf._max_samples,
    )
    sample_blocks = bagging_estimators_sample_indices(
        tuple(int(seed) for seed in clf._seeds),
        clf.bootstrap_features,
        clf.bootstrap,
        clf.n_features_in_,
        clf._n_samples,
        clf._max_features,
        clf._max_samples,
    )
    expected_pairs = tuple(clf._get_estimators_indices())
    expected_feature_blocks = tuple(np.asarray(pair[0], dtype=np.int64) for pair in expected_pairs)
    expected_sample_blocks = tuple(np.asarray(samples, dtype=np.int64) for samples in clf.estimators_samples_)

    for observed, expected in zip(feature_blocks, expected_feature_blocks):
        assert np.array_equal(observed, expected)
    for observed, expected in zip(sample_blocks, expected_sample_blocks):
        assert np.array_equal(observed, expected)


def test_contracts_reject_invalid_bagging_index_reconstruction_inputs() -> None:
    from sciona.atoms.ml.sklearn.ensemble.bagging_index_reconstruction import (
        bagging_estimator_index_pairs,
        bagging_estimators_feature_indices,
        bagging_estimators_sample_indices,
    )

    with pytest.raises(ViolationError):
        bagging_estimator_index_pairs((1, -1), True, True, 5, 7, 3, 4)

    with pytest.raises(ViolationError):
        bagging_estimators_feature_indices((1, 2), False, True, 3, 5, 4, 2)

    with pytest.raises(ViolationError):
        bagging_estimators_sample_indices((1, 2), True, False, 5, 3, 2, 4)
