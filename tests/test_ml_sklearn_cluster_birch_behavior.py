from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError
from sklearn.cluster import Birch


def _data() -> np.ndarray:
    return np.array(
        [
            [0.0, 0.0],
            [0.0, 0.1],
            [5.0, 5.0],
            [5.1, 5.0],
            [9.0, 9.0],
        ],
        dtype=np.float64,
    )


def test_birch_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.cluster.birch import (
        BirchNoGlobalState,
        birch_fit_no_global,
        birch_predict_no_global,
        birch_transform_no_global,
    )

    assert BirchNoGlobalState is not None
    assert callable(birch_fit_no_global)
    assert callable(birch_predict_no_global)
    assert callable(birch_transform_no_global)


def test_birch_fit_no_global_matches_sklearn() -> None:
    from sciona.atoms.ml.sklearn.cluster.birch import birch_fit_no_global

    X = _data()
    state = birch_fit_no_global(X, threshold=0.25, branching_factor=3)
    expected = Birch(threshold=0.25, branching_factor=3, n_clusters=None, compute_labels=True).fit(X)

    assert np.allclose(state.subcluster_centers, expected.subcluster_centers_)
    assert np.array_equal(state.subcluster_labels, expected.subcluster_labels_)
    assert np.array_equal(state.labels, expected.labels_)
    assert state.n_features_in == expected.n_features_in_
    assert state.n_features_out == expected._n_features_out


def test_birch_no_global_predict_and_transform_match_sklearn() -> None:
    from sciona.atoms.ml.sklearn.cluster.birch import (
        birch_fit_no_global,
        birch_predict_no_global,
        birch_transform_no_global,
    )

    X = _data()
    query = np.array([[0.0, 0.05], [5.2, 5.0], [8.0, 8.0]], dtype=np.float64)
    state = birch_fit_no_global(X, threshold=0.25, branching_factor=3)
    expected = Birch(threshold=0.25, branching_factor=3, n_clusters=None, compute_labels=True).fit(X)

    assert np.array_equal(birch_predict_no_global(query, state), expected.predict(query))
    assert np.allclose(birch_transform_no_global(query, state), expected.transform(query))


def test_birch_no_global_compute_labels_false_matches_sklearn_state() -> None:
    from sciona.atoms.ml.sklearn.cluster.birch import birch_fit_no_global, birch_predict_no_global

    X = _data()
    state = birch_fit_no_global(X, threshold=0.25, branching_factor=3, compute_labels=False)
    expected = Birch(threshold=0.25, branching_factor=3, n_clusters=None, compute_labels=False).fit(X)

    assert state.labels is None
    assert np.allclose(state.subcluster_centers, expected.subcluster_centers_)
    assert np.array_equal(state.subcluster_labels, expected.subcluster_labels_)
    assert np.array_equal(birch_predict_no_global(X, state), expected.predict(X))


def test_birch_contracts_reject_invalid_inputs() -> None:
    from sciona.atoms.ml.sklearn.cluster.birch import (
        birch_fit_no_global,
        birch_predict_no_global,
        birch_transform_no_global,
    )

    X = _data()
    state = birch_fit_no_global(X, threshold=0.25, branching_factor=3)

    with pytest.raises(ViolationError):
        birch_fit_no_global(X, threshold=0.0)

    with pytest.raises(ViolationError):
        birch_fit_no_global(X, branching_factor=1)

    with pytest.raises(ViolationError):
        birch_fit_no_global(np.array([[np.nan, 0.0]], dtype=np.float64))

    with pytest.raises(ViolationError):
        birch_predict_no_global(np.ones((2, 3), dtype=np.float64), state)

    with pytest.raises(ViolationError):
        birch_transform_no_global(np.ones((2, 3), dtype=np.float64), state)
