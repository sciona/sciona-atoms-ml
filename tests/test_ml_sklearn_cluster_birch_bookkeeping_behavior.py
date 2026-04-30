from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError
from sklearn.cluster import Birch
from sklearn.exceptions import ConvergenceWarning


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


def test_birch_bookkeeping_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.cluster.birch_bookkeeping import (
        birch_compute_labels_required,
        birch_copy_warning_required,
        birch_first_call,
        birch_identity_subcluster_labels,
        birch_leaf_centers,
        birch_n_features_out,
        birch_not_enough_centroids,
    )

    assert callable(birch_first_call)
    assert callable(birch_copy_warning_required)
    assert callable(birch_compute_labels_required)
    assert callable(birch_not_enough_centroids)
    assert callable(birch_identity_subcluster_labels)
    assert callable(birch_leaf_centers)
    assert callable(birch_n_features_out)


def test_birch_first_call_and_warning_gates_match_private_fit_shell() -> None:
    from sciona.atoms.ml.sklearn.cluster.birch_bookkeeping import (
        birch_copy_warning_required,
        birch_first_call,
    )

    assert birch_first_call(partial=False, has_root=False) is True
    assert birch_first_call(partial=True, has_root=False) is True
    assert birch_first_call(partial=True, has_root=True) is False

    assert birch_copy_warning_required("deprecated", True) is False
    assert birch_copy_warning_required(False, True) is True
    assert birch_copy_warning_required(False, False) is False


def test_birch_leaf_centers_and_n_features_out_match_sklearn_fit_summary() -> None:
    from sciona.atoms.ml.sklearn.cluster.birch_bookkeeping import (
        birch_leaf_centers,
        birch_n_features_out,
    )

    X = _data()
    model = Birch(threshold=0.25, branching_factor=3, n_clusters=None).fit(X)
    leaf_blocks = tuple(np.asarray(leaf.centroids_, dtype=np.float64).copy() for leaf in model._get_leaves())

    actual_centers = birch_leaf_centers(leaf_blocks)

    assert np.allclose(actual_centers, model.subcluster_centers_)
    assert birch_n_features_out(actual_centers) == model._n_features_out


def test_birch_global_noop_bookkeeping_matches_sklearn_none_or_short_circuit() -> None:
    from sciona.atoms.ml.sklearn.cluster.birch_bookkeeping import (
        birch_compute_labels_required,
        birch_identity_subcluster_labels,
        birch_not_enough_centroids,
    )

    X = _data()

    none_model = Birch(threshold=0.25, branching_factor=3, n_clusters=None, compute_labels=True).fit(X)
    assert birch_compute_labels_required(has_input_data=True, compute_labels=True) is True
    assert np.array_equal(
        birch_identity_subcluster_labels(len(none_model.subcluster_centers_)),
        none_model.subcluster_labels_,
    )

    with pytest.warns(ConvergenceWarning):
        short_model = Birch(threshold=0.25, branching_factor=3, n_clusters=10, compute_labels=True).fit(X)

    assert birch_not_enough_centroids(len(short_model.subcluster_centers_), 10) is True
    assert np.array_equal(
        birch_identity_subcluster_labels(len(short_model.subcluster_centers_)),
        short_model.subcluster_labels_,
    )


def test_birch_bookkeeping_contracts_reject_invalid_inputs() -> None:
    from sciona.atoms.ml.sklearn.cluster.birch_bookkeeping import (
        birch_copy_warning_required,
        birch_identity_subcluster_labels,
        birch_leaf_centers,
        birch_n_features_out,
        birch_not_enough_centroids,
    )

    with pytest.raises(ViolationError):
        birch_copy_warning_required("stale", True)

    with pytest.raises(ViolationError):
        birch_not_enough_centroids(0, 2)

    with pytest.raises(ViolationError):
        birch_identity_subcluster_labels(0)

    with pytest.raises(ViolationError):
        birch_leaf_centers((np.ones((1, 2), dtype=np.float64), np.ones((1, 3), dtype=np.float64)))

    with pytest.raises(ViolationError):
        birch_n_features_out(np.ones((0, 2), dtype=np.float64))
