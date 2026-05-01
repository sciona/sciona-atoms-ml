from __future__ import annotations

import numpy as np
import pytest


def test_hdbscan_weighted_centers_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.cluster.hdbscan_weighted_centers import (
        hdbscan_center_cluster_count,
        hdbscan_center_data,
        hdbscan_center_mask,
        hdbscan_center_strength,
        hdbscan_centroid,
        hdbscan_make_centroids,
        hdbscan_make_medoids,
    )

    assert callable(hdbscan_center_cluster_count)
    assert callable(hdbscan_make_centroids)
    assert callable(hdbscan_make_medoids)
    assert callable(hdbscan_center_mask)
    assert callable(hdbscan_center_data)
    assert callable(hdbscan_center_strength)
    assert callable(hdbscan_centroid)


def test_hdbscan_weighted_center_bookkeeping_matches_source_logic() -> None:
    from sciona.atoms.ml.sklearn.cluster.hdbscan_weighted_centers import (
        hdbscan_center_cluster_count,
        hdbscan_center_data,
        hdbscan_center_mask,
        hdbscan_center_strength,
        hdbscan_centroid,
        hdbscan_make_centroids,
        hdbscan_make_medoids,
    )

    labels = np.array([0, 1, 0, 1, -1, -2, -3], dtype=np.int32)
    probabilities = np.array([1.0, 0.2, 0.5, 0.8, 0.0, 0.0, 0.4], dtype=np.float64)
    X = np.array(
        [
            [0.0, 0.0],
            [2.0, 2.0],
            [1.0, 1.0],
            [3.0, 3.0],
            [9.0, 9.0],
            [8.0, 8.0],
            [7.0, 7.0],
        ],
        dtype=np.float64,
    )

    assert hdbscan_center_cluster_count(labels) == 3
    assert hdbscan_make_centroids("centroid") is True
    assert hdbscan_make_centroids("both") is True
    assert hdbscan_make_centroids("medoid") is False
    assert hdbscan_make_medoids("medoid") is True
    assert hdbscan_make_medoids("both") is True
    assert hdbscan_make_medoids("centroid") is False

    mask = hdbscan_center_mask(labels, 0)
    data = hdbscan_center_data(X, mask)
    strength = hdbscan_center_strength(probabilities, mask)
    observed = hdbscan_centroid(data, strength)
    expected = np.average(X[labels == 0], weights=probabilities[labels == 0], axis=0)

    assert np.array_equal(mask, np.array([True, False, True, False, False, False, False]))
    assert np.array_equal(data, X[[0, 2]])
    assert np.array_equal(strength, probabilities[[0, 2]])
    assert np.allclose(observed, expected)


def test_hdbscan_weighted_centers_contracts_reject_invalid_inputs() -> None:
    from sciona.atoms.ml.sklearn.cluster.hdbscan_weighted_centers import (
        hdbscan_center_data,
        hdbscan_centroid,
        hdbscan_make_centroids,
    )

    with pytest.raises(Exception):
        hdbscan_make_centroids("invalid")

    with pytest.raises(Exception):
        hdbscan_center_data(
            np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float64),
            np.array([False, False], dtype=np.bool_),
        )

    with pytest.raises(Exception):
        hdbscan_centroid(
            np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float64),
            np.array([0.0, 0.0], dtype=np.float64),
        )
