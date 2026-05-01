from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import pairwise_distances


def test_hdbscan_weighted_medoids_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.cluster.hdbscan_weighted_medoids import (
        hdbscan_medoid,
        hdbscan_medoid_index,
        hdbscan_medoid_weighted_distance_sums,
        hdbscan_medoid_weighted_distances,
    )

    assert callable(hdbscan_medoid_weighted_distances)
    assert callable(hdbscan_medoid_weighted_distance_sums)
    assert callable(hdbscan_medoid_index)
    assert callable(hdbscan_medoid)


def test_hdbscan_weighted_medoids_match_source_logic() -> None:
    from sciona.atoms.ml.sklearn.cluster.hdbscan_weighted_medoids import (
        hdbscan_medoid,
        hdbscan_medoid_index,
        hdbscan_medoid_weighted_distance_sums,
        hdbscan_medoid_weighted_distances,
    )

    data = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 1.0],
        ],
        dtype=np.float64,
    )
    strength = np.array([1.0, 0.25, 0.75], dtype=np.float64)
    distance_matrix = pairwise_distances(data, metric="euclidean")

    observed_weighted = hdbscan_medoid_weighted_distances(distance_matrix, strength)
    observed_sums = hdbscan_medoid_weighted_distance_sums(observed_weighted)
    observed_index = hdbscan_medoid_index(observed_sums)
    observed_medoid = hdbscan_medoid(data, observed_index)

    expected_weighted = distance_matrix * strength
    expected_sums = expected_weighted.sum(axis=1)
    expected_index = int(np.argmin(expected_sums))
    expected_medoid = data[expected_index]

    assert np.allclose(observed_weighted, expected_weighted)
    assert np.allclose(observed_sums, expected_sums)
    assert observed_index == expected_index
    assert np.allclose(observed_medoid, expected_medoid)


def test_hdbscan_weighted_medoids_contracts_reject_invalid_inputs() -> None:
    from sciona.atoms.ml.sklearn.cluster.hdbscan_weighted_medoids import (
        hdbscan_medoid,
        hdbscan_medoid_weighted_distances,
    )

    with pytest.raises(Exception):
        hdbscan_medoid_weighted_distances(
            np.array([[0.0, 1.0, 2.0]], dtype=np.float64),
            np.array([1.0], dtype=np.float64),
        )

    with pytest.raises(Exception):
        hdbscan_medoid(
            np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float64),
            2,
        )
