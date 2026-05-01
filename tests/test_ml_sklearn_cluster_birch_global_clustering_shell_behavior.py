from __future__ import annotations

import warnings

import numpy as np
import pytest
from icontract import ViolationError
from sklearn.cluster import Birch
from sklearn.exceptions import ConvergenceWarning


def test_birch_global_clustering_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.cluster.birch_global_clustering_shell import (
        birch_global_short_circuit_required,
        birch_not_enough_centroids_warning_message,
        birch_not_enough_centroids_warning_required,
        birch_partial_fit_global_only_required,
    )

    assert callable(birch_partial_fit_global_only_required)
    assert callable(birch_global_short_circuit_required)
    assert callable(birch_not_enough_centroids_warning_required)
    assert callable(birch_not_enough_centroids_warning_message)


def test_birch_global_clustering_shell_matches_sklearn_warning_path() -> None:
    from sciona.atoms.ml.sklearn.cluster.birch_global_clustering_shell import (
        birch_global_short_circuit_required,
        birch_not_enough_centroids_warning_message,
        birch_not_enough_centroids_warning_required,
        birch_partial_fit_global_only_required,
    )
    from sciona.atoms.ml.sklearn.cluster.birch_bookkeeping import (
        birch_compute_labels_required,
        birch_not_enough_centroids,
    )

    X = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [0.0, 0.1],
            [0.1, 0.1],
        ],
        dtype=np.float64,
    )

    model = Birch(threshold=100.0, n_clusters=4, compute_labels=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(X)

    warning_messages = [
        str(item.message)
        for item in caught
        if issubclass(item.category, ConvergenceWarning)
    ]
    assert len(warning_messages) == 1

    n_centroids = int(model.subcluster_centers_.shape[0])
    not_enough = birch_not_enough_centroids(n_centroids, 4)
    assert not_enough is True
    assert birch_partial_fit_global_only_required(False) is True
    assert birch_compute_labels_required(True, True) is True
    assert birch_global_short_circuit_required(False, not_enough) is True
    assert birch_not_enough_centroids_warning_required(not_enough) is True
    assert birch_not_enough_centroids_warning_message(n_centroids, 4) == warning_messages[0]


def test_birch_global_clustering_shell_no_warning_branch() -> None:
    from sciona.atoms.ml.sklearn.cluster.birch_global_clustering_shell import (
        birch_global_short_circuit_required,
        birch_not_enough_centroids_warning_required,
        birch_partial_fit_global_only_required,
    )

    assert birch_partial_fit_global_only_required(True) is False
    assert birch_global_short_circuit_required(True, False) is True
    assert birch_global_short_circuit_required(False, False) is False
    assert birch_not_enough_centroids_warning_required(False) is False


def test_birch_global_clustering_shell_contracts_reject_invalid_inputs() -> None:
    from sciona.atoms.ml.sklearn.cluster.birch_global_clustering_shell import (
        birch_global_short_circuit_required,
        birch_not_enough_centroids_warning_message,
        birch_not_enough_centroids_warning_required,
        birch_partial_fit_global_only_required,
    )

    with pytest.raises(ViolationError):
        birch_partial_fit_global_only_required(1)

    with pytest.raises(ViolationError):
        birch_global_short_circuit_required(False, 0)

    with pytest.raises(ViolationError):
        birch_not_enough_centroids_warning_required("no")

    with pytest.raises(ViolationError):
        birch_not_enough_centroids_warning_message(0, 2)
