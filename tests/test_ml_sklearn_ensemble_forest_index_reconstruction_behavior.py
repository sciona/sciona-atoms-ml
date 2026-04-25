from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier


def test_forest_index_reconstruction_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.ensemble.forest_index_reconstruction import (
        forest_estimator_sample_indices,
        forest_estimators_sample_indices,
    )

    assert callable(forest_estimator_sample_indices)
    assert callable(forest_estimators_sample_indices)


def test_forest_estimator_sample_indices_match_sklearn_bootstrap_tree() -> None:
    from sciona.atoms.ml.sklearn.ensemble.forest_index_reconstruction import (
        forest_estimator_sample_indices,
    )

    X, y = make_classification(
        n_samples=30,
        n_features=5,
        n_informative=4,
        n_redundant=0,
        random_state=7,
    )
    clf = RandomForestClassifier(
        n_estimators=4,
        bootstrap=True,
        max_samples=9,
        random_state=3,
    ).fit(X, y)

    expected = tuple(np.asarray(samples, dtype=np.int64) for samples in clf.estimators_samples_)
    observed = tuple(
        forest_estimator_sample_indices(
            clf.bootstrap,
            clf._n_samples,
            clf._n_samples_bootstrap,
            int(tree.random_state),
        )
        for tree in clf.estimators_
    )

    for obs, exp in zip(observed, expected):
        assert np.array_equal(obs, exp)


def test_forest_estimators_sample_indices_match_sklearn_property() -> None:
    from sciona.atoms.ml.sklearn.ensemble.forest_index_reconstruction import (
        forest_estimators_sample_indices,
    )

    X, y = make_classification(
        n_samples=24,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        random_state=11,
    )
    clf = RandomForestClassifier(
        n_estimators=5,
        bootstrap=True,
        max_samples=0.5,
        random_state=17,
    ).fit(X, y)

    observed = forest_estimators_sample_indices(
        tuple(int(tree.random_state) for tree in clf.estimators_),
        clf.bootstrap,
        clf._n_samples,
        clf._n_samples_bootstrap,
    )
    expected = tuple(np.asarray(samples, dtype=np.int64) for samples in clf.estimators_samples_)

    for obs, exp in zip(observed, expected):
        assert np.array_equal(obs, exp)


def test_forest_estimators_sample_indices_match_nonbootstrap_property() -> None:
    from sciona.atoms.ml.sklearn.ensemble.forest_index_reconstruction import (
        forest_estimators_sample_indices,
    )

    X, y = make_classification(
        n_samples=18,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        random_state=13,
    )
    clf = RandomForestClassifier(
        n_estimators=3,
        bootstrap=False,
        random_state=19,
    ).fit(X, y)

    observed = forest_estimators_sample_indices(
        tuple(int(tree.random_state) for tree in clf.estimators_),
        clf.bootstrap,
        clf._n_samples,
        clf._n_samples_bootstrap,
    )
    expected = tuple(np.asarray(samples, dtype=np.int64) for samples in clf.estimators_samples_)

    for obs, exp in zip(observed, expected):
        assert np.array_equal(obs, exp)
    assert all(np.array_equal(obs, np.arange(clf._n_samples, dtype=np.int64)) for obs in observed)


def test_contracts_reject_invalid_forest_index_reconstruction_inputs() -> None:
    from sciona.atoms.ml.sklearn.ensemble.forest_index_reconstruction import (
        forest_estimator_sample_indices,
        forest_estimators_sample_indices,
    )

    with pytest.raises(ViolationError):
        forest_estimator_sample_indices(True, 10, None, 0)

    with pytest.raises(ViolationError):
        forest_estimator_sample_indices(False, 10, 4, 0)

    with pytest.raises(ViolationError):
        forest_estimators_sample_indices((1, -1), True, 10, 5)
