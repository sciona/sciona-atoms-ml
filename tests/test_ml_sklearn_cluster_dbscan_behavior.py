from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError
from sklearn.cluster import DBSCAN as SklearnDBSCAN
from sklearn.cluster import dbscan as sklearn_dbscan
from sklearn.metrics import pairwise_distances


def _data() -> np.ndarray:
    return np.array(
        [
            [0.0, 0.0],
            [0.0, 0.1],
            [0.2, 0.0],
            [5.0, 5.0],
            [5.1, 5.0],
            [10.0, 10.0],
        ],
        dtype=np.float64,
    )


def test_dbscan_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.cluster.dbscan import DBSCANState, dbscan_core_labels, dbscan_fit

    assert DBSCANState is not None
    assert callable(dbscan_core_labels)
    assert callable(dbscan_fit)


def test_dbscan_fit_matches_sklearn_estimator() -> None:
    from sciona.atoms.ml.sklearn.cluster.dbscan import dbscan_fit

    X = _data()
    state = dbscan_fit(X, eps=0.35, min_samples=2, metric="euclidean")
    expected = SklearnDBSCAN(eps=0.35, min_samples=2, metric="euclidean").fit(X)

    assert np.array_equal(state.core_sample_indices, expected.core_sample_indices_)
    assert np.array_equal(state.labels, expected.labels_)
    assert np.allclose(state.components, expected.components_)
    assert state.eps == expected.eps
    assert state.min_samples == expected.min_samples
    assert state.metric == expected.metric
    assert state.n_features_in == expected.n_features_in_


def test_dbscan_core_labels_matches_sklearn_public_helper() -> None:
    from sciona.atoms.ml.sklearn.cluster.dbscan import dbscan_core_labels

    X = _data()
    weights = np.array([1.0, 1.0, 2.0, 1.0, 1.0, -0.5], dtype=np.float64)
    core_indices, labels = dbscan_core_labels(
        X,
        eps=0.35,
        min_samples=2,
        metric="minkowski",
        p=2.0,
        sample_weight=weights,
    )
    expected_core_indices, expected_labels = sklearn_dbscan(
        X,
        eps=0.35,
        min_samples=2,
        metric="minkowski",
        p=2.0,
        sample_weight=weights,
    )

    assert np.array_equal(core_indices, expected_core_indices)
    assert np.array_equal(labels, expected_labels)


def test_dbscan_precomputed_dense_distances_match_sklearn() -> None:
    from sciona.atoms.ml.sklearn.cluster.dbscan import dbscan_core_labels, dbscan_fit

    distances = pairwise_distances(_data())
    state = dbscan_fit(distances, eps=0.35, min_samples=2, metric="precomputed", algorithm="brute")
    expected = SklearnDBSCAN(eps=0.35, min_samples=2, metric="precomputed", algorithm="brute").fit(distances)
    core_indices, labels = dbscan_core_labels(distances, eps=0.35, min_samples=2, metric="precomputed", algorithm="brute")

    assert np.array_equal(state.labels, expected.labels_)
    assert np.array_equal(state.core_sample_indices, expected.core_sample_indices_)
    assert np.array_equal(core_indices, expected.core_sample_indices_)
    assert np.array_equal(labels, expected.labels_)


def test_contracts_reject_invalid_dbscan_inputs() -> None:
    from sciona.atoms.ml.sklearn.cluster.dbscan import dbscan_core_labels, dbscan_fit

    X = _data()

    with pytest.raises(ViolationError):
        dbscan_fit(X, eps=0.0)

    with pytest.raises(ViolationError):
        dbscan_fit(X, min_samples=0)

    with pytest.raises(ViolationError):
        dbscan_fit(X, metric="cosine")

    with pytest.raises(ViolationError):
        dbscan_fit(X, sample_weight=np.ones(X.shape[0] + 1, dtype=np.float64))

    with pytest.raises(ViolationError):
        dbscan_core_labels(np.ones((2, 3), dtype=np.float64), metric="precomputed")
