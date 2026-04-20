from __future__ import annotations

import numpy as np
import pytest
from sklearn.cluster import OPTICS as SklearnOPTICS


def test_optics_fit_atom_import() -> None:
    from sciona.atoms.ml.sklearn.cluster import OpticsState, optics_fit

    assert callable(optics_fit)
    assert OpticsState is not None


def test_optics_fit_matches_sklearn_xi() -> None:
    from sciona.atoms.ml.sklearn.cluster import optics_fit

    X = np.array(
        [[1, 2], [2, 5], [3, 6], [8, 7], [8, 8], [7, 3], [25, 30], [26, 31]],
        dtype=np.float64,
    )

    state = optics_fit(X, min_samples=2, cluster_method="xi")
    expected = SklearnOPTICS(min_samples=2, cluster_method="xi").fit(X)

    assert np.array_equal(state.ordering, expected.ordering_)
    assert np.allclose(state.core_distances, expected.core_distances_, equal_nan=True)
    assert np.allclose(state.reachability, expected.reachability_, equal_nan=True)
    assert np.array_equal(state.predecessor, expected.predecessor_)
    assert np.array_equal(state.labels, expected.labels_)
    assert np.array_equal(state.cluster_hierarchy, expected.cluster_hierarchy_)
    assert state.cluster_method == "xi"
    assert state.n_features_in == expected.n_features_in_


def test_optics_fit_matches_sklearn_dbscan_extraction() -> None:
    from sciona.atoms.ml.sklearn.cluster import optics_fit

    X = np.array(
        [[1, 2], [2, 5], [3, 6], [8, 7], [8, 8], [7, 3], [25, 30], [26, 31]],
        dtype=np.float64,
    )

    state = optics_fit(X, min_samples=2, cluster_method="dbscan", eps=4.5)
    expected = SklearnOPTICS(min_samples=2, cluster_method="dbscan", eps=4.5).fit(X)

    assert np.array_equal(state.ordering, expected.ordering_)
    assert np.allclose(state.core_distances, expected.core_distances_, equal_nan=True)
    assert np.allclose(state.reachability, expected.reachability_, equal_nan=True)
    assert np.array_equal(state.predecessor, expected.predecessor_)
    assert np.array_equal(state.labels, expected.labels_)
    assert state.cluster_hierarchy is None
    assert state.cluster_method == "dbscan"


def test_optics_fit_rejects_dbscan_eps_greater_than_max_eps() -> None:
    from sciona.atoms.ml.sklearn.cluster import optics_fit

    X = np.array([[0.0, 0.0], [0.0, 1.0], [5.0, 5.0]], dtype=np.float64)

    with pytest.raises(Exception):
        optics_fit(X, min_samples=2, max_eps=1.0, cluster_method="dbscan", eps=2.0)
