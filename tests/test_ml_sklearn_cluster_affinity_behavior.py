from __future__ import annotations

import numpy as np
import pytest
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import affinity_propagation as sklearn_affinity_propagation
from sklearn.metrics.pairwise import euclidean_distances


def test_affinity_cluster_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.cluster import (
        affinity_propagation,
        affinity_propagation_fit,
        affinity_propagation_predict,
    )

    assert callable(affinity_propagation)
    assert callable(affinity_propagation_fit)
    assert callable(affinity_propagation_predict)


def test_affinity_propagation_function_matches_sklearn() -> None:
    from sciona.atoms.ml.sklearn.cluster import affinity_propagation

    X = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]], dtype=np.float64)
    similarities = -euclidean_distances(X, squared=True)

    result = affinity_propagation(similarities, random_state=0, return_n_iter=True)
    expected = sklearn_affinity_propagation(similarities, random_state=0, return_n_iter=True)

    assert np.array_equal(result[0], expected[0])
    assert np.array_equal(result[1], expected[1])
    assert result[2] == expected[2]


def test_affinity_propagation_fit_predict_matches_sklearn_euclidean() -> None:
    from sciona.atoms.ml.sklearn.cluster import affinity_propagation_fit, affinity_propagation_predict

    X = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]], dtype=np.float64)
    query = np.array([[0, 0], [4, 4]], dtype=np.float64)

    state = affinity_propagation_fit(X, random_state=5)
    expected = AffinityPropagation(random_state=5).fit(X)

    assert np.array_equal(state.cluster_centers_indices, expected.cluster_centers_indices_)
    assert np.array_equal(state.labels, expected.labels_)
    assert state.n_iter == expected.n_iter_
    assert np.allclose(state.cluster_centers, expected.cluster_centers_)
    assert np.array_equal(affinity_propagation_predict(query, state), expected.predict(query))


def test_affinity_propagation_precomputed_fit_matches_sklearn() -> None:
    from sciona.atoms.ml.sklearn.cluster import affinity_propagation_fit, affinity_propagation_predict

    X = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]], dtype=np.float64)
    similarities = -euclidean_distances(X, squared=True)

    state = affinity_propagation_fit(similarities, affinity="precomputed", random_state=0)
    expected = AffinityPropagation(affinity="precomputed", random_state=0).fit(similarities)

    assert np.array_equal(state.cluster_centers_indices, expected.cluster_centers_indices_)
    assert np.array_equal(state.labels, expected.labels_)
    assert state.cluster_centers is None
    with pytest.raises(ValueError, match="precomputed"):
        affinity_propagation_predict(X, state)


def test_affinity_propagation_equal_similarity_branch() -> None:
    from sciona.atoms.ml.sklearn.cluster import affinity_propagation

    similarities = np.ones((3, 3), dtype=np.float64)
    with pytest.warns(UserWarning, match="mutually equal similarities"):
        centers, labels, n_iter = affinity_propagation(
            similarities,
            preference=2.0,
            random_state=0,
            return_n_iter=True,
        )

    assert np.array_equal(centers, np.array([0, 1, 2]))
    assert np.array_equal(labels, np.array([0, 1, 2]))
    assert n_iter == 0
