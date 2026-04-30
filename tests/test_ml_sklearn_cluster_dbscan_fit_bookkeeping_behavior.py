from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
from icontract import ViolationError
from sklearn.cluster import DBSCAN as SklearnDBSCAN
from sklearn.neighbors import NearestNeighbors


def _data() -> np.ndarray:
    return np.array(
        [
            [0.0, 0.0],
            [0.0, 0.1],
            [0.2, 0.0],
            [5.0, 5.0],
            [5.1, 5.0],
            [10.0, 10.0],
        ],
        dtype=np.float64,
    )


def test_dbscan_fit_bookkeeping_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.cluster.dbscan_fit_bookkeeping import (
        dbscan_core_sample_mask,
        dbscan_dense_core_components,
        dbscan_empty_components,
        dbscan_initial_noise_labels,
        dbscan_neighbor_count_vector,
        dbscan_precomputed_sparse_self_neighbors,
        dbscan_weighted_neighbor_sums,
    )

    assert callable(dbscan_precomputed_sparse_self_neighbors)
    assert callable(dbscan_neighbor_count_vector)
    assert callable(dbscan_weighted_neighbor_sums)
    assert callable(dbscan_core_sample_mask)
    assert callable(dbscan_initial_noise_labels)
    assert callable(dbscan_dense_core_components)
    assert callable(dbscan_empty_components)


def test_dbscan_sparse_precomputed_self_neighbors_matches_sklearn_branch() -> None:
    from sciona.atoms.ml.sklearn.cluster.dbscan_fit_bookkeeping import dbscan_precomputed_sparse_self_neighbors

    distances = np.array(
        [
            [0.0, 0.2, 0.0],
            [0.2, 0.0, 0.3],
            [0.0, 0.3, 0.0],
        ],
        dtype=np.float64,
    )
    graph = sp.csr_matrix(distances)

    expected = graph.copy()
    expected.setdiag(expected.diagonal())
    actual = dbscan_precomputed_sparse_self_neighbors(graph)

    assert sp.isspmatrix_csr(actual)
    assert np.array_equal(actual.diagonal(), expected.diagonal())
    assert np.array_equal(actual.toarray(), expected.toarray())


def test_dbscan_neighbor_reductions_and_core_mask_match_sklearn_fit_shell() -> None:
    from sciona.atoms.ml.sklearn.cluster.dbscan_fit_bookkeeping import (
        dbscan_core_sample_mask,
        dbscan_neighbor_count_vector,
        dbscan_weighted_neighbor_sums,
    )

    X = _data()
    model = NearestNeighbors(radius=0.35).fit(X)
    neighborhoods = tuple(model.radius_neighbors(X, return_distance=False))
    sample_weight = np.array([1.0, 1.0, 2.0, 1.0, 1.0, -0.5], dtype=np.float64)

    expected_counts = np.array([len(neighbors) for neighbors in neighborhoods], dtype=np.int64)
    expected_weight_sums = np.array([np.sum(sample_weight[neighbors]) for neighbors in neighborhoods], dtype=np.float64)

    actual_counts = dbscan_neighbor_count_vector(neighborhoods)
    actual_weight_sums = dbscan_weighted_neighbor_sums(neighborhoods, sample_weight)

    assert np.array_equal(actual_counts, expected_counts)
    assert np.allclose(actual_weight_sums, expected_weight_sums)
    assert np.array_equal(dbscan_core_sample_mask(actual_counts, 2), np.asarray(expected_counts >= 2, dtype=np.uint8))
    assert np.array_equal(dbscan_core_sample_mask(actual_weight_sums, 2), np.asarray(expected_weight_sums >= 2, dtype=np.uint8))


def test_dbscan_label_and_component_packaging_matches_sklearn() -> None:
    from sciona.atoms.ml.sklearn.cluster.dbscan_fit_bookkeeping import (
        dbscan_dense_core_components,
        dbscan_empty_components,
        dbscan_initial_noise_labels,
    )

    X = _data()
    expected = SklearnDBSCAN(eps=0.35, min_samples=2, metric="euclidean").fit(X)

    assert np.array_equal(dbscan_initial_noise_labels(X.shape[0]), np.full(X.shape[0], -1, dtype=np.intp))
    assert np.allclose(
        dbscan_dense_core_components(X, expected.core_sample_indices_.astype(np.intp)),
        expected.components_,
    )
    assert dbscan_empty_components(X.shape[1]).shape == (0, X.shape[1])


def test_dbscan_fit_bookkeeping_contracts_reject_invalid_inputs() -> None:
    from sciona.atoms.ml.sklearn.cluster.dbscan_fit_bookkeeping import (
        dbscan_core_sample_mask,
        dbscan_dense_core_components,
        dbscan_empty_components,
        dbscan_initial_noise_labels,
        dbscan_neighbor_count_vector,
        dbscan_precomputed_sparse_self_neighbors,
        dbscan_weighted_neighbor_sums,
    )

    with pytest.raises(ViolationError):
        dbscan_precomputed_sparse_self_neighbors(np.eye(2, dtype=np.float64))

    with pytest.raises(ViolationError):
        dbscan_neighbor_count_vector(())

    with pytest.raises(ViolationError):
        dbscan_weighted_neighbor_sums((np.array([0, 1], dtype=np.intp),), np.array([1.0, 2.0], dtype=np.float64))

    with pytest.raises(ViolationError):
        dbscan_core_sample_mask(np.array([], dtype=np.float64), 2)

    with pytest.raises(ViolationError):
        dbscan_initial_noise_labels(0)

    with pytest.raises(ViolationError):
        dbscan_dense_core_components(np.array([[0.0, 1.0]], dtype=np.float64), np.array([1], dtype=np.intp))

    with pytest.raises(ViolationError):
        dbscan_empty_components(0)
