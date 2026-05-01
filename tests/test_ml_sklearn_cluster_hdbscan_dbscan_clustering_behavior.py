from __future__ import annotations

import numpy as np
import pytest
from sklearn.cluster import HDBSCAN
from sklearn.cluster._hdbscan.hdbscan import labelling_at_cut


def _finite_data() -> np.ndarray:
    return np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [5.0, 5.0],
            [5.1, 5.0],
        ],
        dtype=np.float64,
    )


def _nonfinite_data() -> np.ndarray:
    return np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [5.0, 5.0],
            [5.1, 5.0],
            [np.inf, 0.0],
            [np.nan, 1.0],
        ],
        dtype=np.float64,
    )


def test_hdbscan_dbscan_clustering_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.cluster.hdbscan_dbscan_clustering import (
        hdbscan_dbscan_infinite_mask,
        hdbscan_dbscan_labels,
        hdbscan_dbscan_missing_mask,
    )

    assert callable(hdbscan_dbscan_infinite_mask)
    assert callable(hdbscan_dbscan_missing_mask)
    assert callable(hdbscan_dbscan_labels)


def test_hdbscan_dbscan_clustering_matches_sklearn_for_finite_data() -> None:
    from sciona.atoms.ml.sklearn.cluster.hdbscan_dbscan_clustering import (
        hdbscan_dbscan_infinite_mask,
        hdbscan_dbscan_labels,
        hdbscan_dbscan_missing_mask,
    )

    model = HDBSCAN(min_cluster_size=2).fit(_finite_data())
    labels_at_cut = labelling_at_cut(model._single_linkage_tree_, 0.5, 2)

    observed = hdbscan_dbscan_labels(
        labels_at_cut,
        hdbscan_dbscan_infinite_mask(model.labels_),
        hdbscan_dbscan_missing_mask(model.labels_),
    )

    expected = model.dbscan_clustering(0.5, 2)
    assert np.array_equal(observed, expected)


def test_hdbscan_dbscan_clustering_matches_sklearn_for_nonfinite_data() -> None:
    from sciona.atoms.ml.sklearn.cluster.hdbscan_dbscan_clustering import (
        hdbscan_dbscan_infinite_mask,
        hdbscan_dbscan_labels,
        hdbscan_dbscan_missing_mask,
    )

    model = HDBSCAN(min_cluster_size=2).fit(_nonfinite_data())
    labels_at_cut = labelling_at_cut(model._single_linkage_tree_, 0.5, 2)

    infinite_mask = hdbscan_dbscan_infinite_mask(model.labels_)
    missing_mask = hdbscan_dbscan_missing_mask(model.labels_)
    observed = hdbscan_dbscan_labels(labels_at_cut, infinite_mask, missing_mask)

    expected = model.dbscan_clustering(0.5, 2)
    assert np.array_equal(infinite_mask, model.labels_ == -2)
    assert np.array_equal(missing_mask, model.labels_ == -3)
    assert np.array_equal(observed, expected)
    assert observed[-2] == -2
    assert observed[-1] == -3


def test_hdbscan_dbscan_clustering_contracts_reject_invalid_inputs() -> None:
    from sciona.atoms.ml.sklearn.cluster.hdbscan_dbscan_clustering import (
        hdbscan_dbscan_labels,
        hdbscan_dbscan_missing_mask,
    )

    with pytest.raises(Exception):
        hdbscan_dbscan_missing_mask(np.array([[0, -3]], dtype=np.int32))

    with pytest.raises(Exception):
        hdbscan_dbscan_labels(
            np.array([0, 1], dtype=np.int32),
            np.array([True], dtype=np.bool_),
            np.array([False, False], dtype=np.bool_),
        )
