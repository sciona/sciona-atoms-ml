from __future__ import annotations

import numpy as np
import pytest
from sklearn.cluster import compute_optics_graph as sklearn_compute_optics_graph
from sklearn.metrics import pairwise_distances


def test_optics_graph_atom_import() -> None:
    from sciona.atoms.ml.sklearn.cluster import compute_optics_graph

    assert callable(compute_optics_graph)


def test_compute_optics_graph_matches_sklearn_dense() -> None:
    from sciona.atoms.ml.sklearn.cluster import compute_optics_graph

    X = np.array([[1, 2], [2, 5], [3, 6], [8, 7], [8, 8], [7, 3]], dtype=np.float64)
    result = compute_optics_graph(
        X,
        min_samples=2,
        max_eps=np.inf,
        metric="minkowski",
        p=2,
        metric_params=None,
        algorithm="auto",
        leaf_size=30,
        n_jobs=None,
    )
    expected = sklearn_compute_optics_graph(
        X,
        min_samples=2,
        max_eps=np.inf,
        metric="minkowski",
        p=2,
        metric_params=None,
        algorithm="auto",
        leaf_size=30,
        n_jobs=None,
    )

    assert np.array_equal(result[0], expected[0])
    assert np.allclose(result[1], expected[1])
    assert np.allclose(result[2], expected[2])
    assert np.array_equal(result[3], expected[3])


def test_compute_optics_graph_matches_sklearn_precomputed() -> None:
    from sciona.atoms.ml.sklearn.cluster import compute_optics_graph

    X = np.array([[1, 2], [2, 5], [3, 6], [8, 7], [8, 8], [7, 3]], dtype=np.float64)
    distances = pairwise_distances(X)
    result = compute_optics_graph(
        distances,
        min_samples=2,
        max_eps=np.inf,
        metric="precomputed",
        p=None,
        metric_params=None,
        algorithm="brute",
        leaf_size=30,
        n_jobs=None,
    )
    expected = sklearn_compute_optics_graph(
        distances,
        min_samples=2,
        max_eps=np.inf,
        metric="precomputed",
        p=None,
        metric_params=None,
        algorithm="brute",
        leaf_size=30,
        n_jobs=None,
    )

    assert np.array_equal(result[0], expected[0])
    assert np.allclose(result[1], expected[1])
    assert np.allclose(result[2], expected[2])
    assert np.array_equal(result[3], expected[3])


def test_compute_optics_graph_warns_when_all_unreachable() -> None:
    from sciona.atoms.ml.sklearn.cluster import compute_optics_graph

    X = np.array([[0, 0], [100, 100], [200, 200]], dtype=np.float64)
    with pytest.warns(UserWarning, match="All reachability values are inf"):
        ordering, core_distances, reachability, predecessor = compute_optics_graph(
            X,
            min_samples=2,
            max_eps=1.0,
            metric="minkowski",
            p=2,
            metric_params=None,
            algorithm="auto",
            leaf_size=30,
            n_jobs=None,
        )

    assert np.array_equal(ordering, np.array([0, 1, 2]))
    assert np.all(np.isinf(core_distances))
    assert np.all(np.isinf(reachability))
    assert np.array_equal(predecessor, np.array([-1, -1, -1]))
