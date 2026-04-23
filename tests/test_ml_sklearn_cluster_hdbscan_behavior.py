from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError
from sklearn.cluster import HDBSCAN as SklearnHDBSCAN


def _data() -> np.ndarray:
    return np.array(
        [
            [0.0, 0.0],
            [0.0, 0.1],
            [0.1, 0.0],
            [5.0, 5.0],
            [5.1, 5.0],
            [5.0, 5.1],
            [10.0, 10.0],
        ],
        dtype=np.float64,
    )


def test_hdbscan_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.cluster.hdbscan import HDBSCANState, hdbscan_fit, hdbscan_fit_predict

    assert HDBSCANState is not None
    assert callable(hdbscan_fit)
    assert callable(hdbscan_fit_predict)


def test_hdbscan_fit_matches_sklearn_brute() -> None:
    from sciona.atoms.ml.sklearn.cluster.hdbscan import hdbscan_fit

    X = _data()
    state = hdbscan_fit(X, min_cluster_size=2, min_samples=2, algorithm="brute", metric="euclidean")
    expected = SklearnHDBSCAN(min_cluster_size=2, min_samples=2, algorithm="brute", metric="euclidean", copy=True).fit(X)

    assert np.array_equal(state.labels, expected.labels_)
    assert np.allclose(state.probabilities, expected.probabilities_)
    assert np.array_equal(state.single_linkage_tree, expected._single_linkage_tree_)
    assert state.min_samples == expected._min_samples
    assert state.n_features_in == expected.n_features_in_


def test_hdbscan_fit_matches_sklearn_auto_leaf_selection() -> None:
    from sciona.atoms.ml.sklearn.cluster.hdbscan import hdbscan_fit

    X = _data()
    state = hdbscan_fit(
        X,
        min_cluster_size=2,
        min_samples=None,
        algorithm="auto",
        metric="manhattan",
        cluster_selection_method="leaf",
        allow_single_cluster=True,
    )
    expected = SklearnHDBSCAN(
        min_cluster_size=2,
        min_samples=None,
        algorithm="auto",
        metric="manhattan",
        cluster_selection_method="leaf",
        allow_single_cluster=True,
        copy=True,
    ).fit(X)

    assert np.array_equal(state.labels, expected.labels_)
    assert np.allclose(state.probabilities, expected.probabilities_)
    assert np.array_equal(state.single_linkage_tree, expected._single_linkage_tree_)
    assert state.min_samples == expected._min_samples
    assert state.cluster_selection_method == "leaf"
    assert state.allow_single_cluster is True


def test_hdbscan_fit_predict_matches_sklearn() -> None:
    from sciona.atoms.ml.sklearn.cluster.hdbscan import hdbscan_fit_predict

    X = _data()
    labels = hdbscan_fit_predict(X, min_cluster_size=2, min_samples=2, algorithm="brute", metric="euclidean")
    expected = SklearnHDBSCAN(min_cluster_size=2, min_samples=2, algorithm="brute", metric="euclidean", copy=True).fit_predict(X)

    assert np.array_equal(labels, expected)


def test_hdbscan_contracts_reject_invalid_inputs() -> None:
    from sciona.atoms.ml.sklearn.cluster.hdbscan import hdbscan_fit, hdbscan_fit_predict

    X = _data()

    with pytest.raises(ViolationError):
        hdbscan_fit(X[:1], min_cluster_size=2)

    with pytest.raises(ViolationError):
        hdbscan_fit(X, min_cluster_size=1)

    with pytest.raises(ViolationError):
        hdbscan_fit(X, min_samples=0)

    with pytest.raises(ViolationError):
        hdbscan_fit(X, metric="precomputed")

    with pytest.raises(ViolationError):
        hdbscan_fit(X, alpha=0.0)

    with pytest.raises(ViolationError):
        hdbscan_fit_predict(X, cluster_selection_method="middle")
