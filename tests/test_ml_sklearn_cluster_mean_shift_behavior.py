from __future__ import annotations

import numpy as np
from sklearn.cluster import MeanShift
from sklearn.cluster import estimate_bandwidth as sklearn_estimate_bandwidth
from sklearn.cluster import mean_shift as sklearn_mean_shift


def test_mean_shift_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.cluster import (
        estimate_bandwidth,
        mean_shift,
        mean_shift_fit,
        mean_shift_predict,
    )

    assert callable(estimate_bandwidth)
    assert callable(mean_shift)
    assert callable(mean_shift_fit)
    assert callable(mean_shift_predict)


def test_estimate_bandwidth_matches_sklearn() -> None:
    from sciona.atoms.ml.sklearn.cluster import estimate_bandwidth

    X = np.array([[1, 1], [2, 1], [1, 0], [4, 7], [3, 5], [3, 6]], dtype=np.float64)
    result = estimate_bandwidth(X, quantile=0.5, random_state=0)
    expected = sklearn_estimate_bandwidth(X, quantile=0.5, random_state=0)
    assert np.isclose(result, expected)


def test_mean_shift_matches_sklearn_function() -> None:
    from sciona.atoms.ml.sklearn.cluster import mean_shift

    X = np.array([[1, 1], [2, 1], [1, 0], [4, 7], [3, 5], [3, 6]], dtype=np.float64)
    centers, labels = mean_shift(X, bandwidth=2.0)
    expected_centers, expected_labels = sklearn_mean_shift(X, bandwidth=2.0)

    assert np.allclose(centers, expected_centers)
    assert np.array_equal(labels, expected_labels)


def test_mean_shift_fit_predict_matches_sklearn_estimator() -> None:
    from sciona.atoms.ml.sklearn.cluster import mean_shift_fit, mean_shift_predict

    X = np.array([[1, 1], [2, 1], [1, 0], [4, 7], [3, 5], [3, 6]], dtype=np.float64)
    query = np.array([[1, 1], [4, 6]], dtype=np.float64)

    state = mean_shift_fit(X, bandwidth=2.0)
    expected = MeanShift(bandwidth=2.0).fit(X)

    assert np.allclose(state.cluster_centers, expected.cluster_centers_)
    assert np.array_equal(state.labels, expected.labels_)
    assert state.n_iter == expected.n_iter_
    assert np.array_equal(mean_shift_predict(query, state), expected.predict(query))


def test_mean_shift_cluster_all_false_matches_sklearn() -> None:
    from sciona.atoms.ml.sklearn.cluster import mean_shift_fit

    X = np.array([[0, 0], [0, 0.1], [10, 10]], dtype=np.float64)
    seeds = np.array([[0, 0]], dtype=np.float64)
    state = mean_shift_fit(X, bandwidth=0.5, seeds=seeds, cluster_all=False)
    expected = MeanShift(bandwidth=0.5, seeds=seeds, cluster_all=False).fit(X)

    assert np.allclose(state.cluster_centers, expected.cluster_centers_)
    assert np.array_equal(state.labels, expected.labels_)
    assert -1 in state.labels


def test_mean_shift_bin_seeding_matches_sklearn() -> None:
    from sciona.atoms.ml.sklearn.cluster import mean_shift_fit

    X = np.array([[1, 1], [2, 1], [1, 0], [4, 7], [3, 5], [3, 6]], dtype=np.float64)
    state = mean_shift_fit(X, bandwidth=2.0, bin_seeding=True, min_bin_freq=1)
    expected = MeanShift(bandwidth=2.0, bin_seeding=True, min_bin_freq=1).fit(X)

    assert np.allclose(state.cluster_centers, expected.cluster_centers_)
    assert np.array_equal(state.labels, expected.labels_)
