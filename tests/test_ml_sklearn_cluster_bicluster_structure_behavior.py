from __future__ import annotations

from types import MethodType

import numpy as np
import pytest
from sklearn.cluster import SpectralBiclustering, SpectralCoclustering


def test_bicluster_structure_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.cluster.bicluster_structure import (
        bicluster_effective_svd_dims,
        bicluster_indicator_grid,
        bicluster_resolve_cluster_counts,
        cocluster_indicator_matrix,
        cocluster_singular_vector_count,
        cocluster_split_labels,
        cocluster_stacked_embedding,
    )

    assert callable(cocluster_singular_vector_count)
    assert callable(cocluster_stacked_embedding)
    assert callable(cocluster_split_labels)
    assert callable(cocluster_indicator_matrix)
    assert callable(bicluster_effective_svd_dims)
    assert callable(bicluster_resolve_cluster_counts)
    assert callable(bicluster_indicator_grid)


def test_cocluster_structure_helpers_match_private_fit_bookkeeping() -> None:
    from sciona.atoms.ml.sklearn.cluster.bicluster import bicluster_scale_normalize
    from sciona.atoms.ml.sklearn.cluster.bicluster_structure import (
        cocluster_indicator_matrix,
        cocluster_singular_vector_count,
        cocluster_split_labels,
        cocluster_stacked_embedding,
    )

    X = np.array(
        [
            [1.0, 2.0, 0.0],
            [3.0, 1.0, 2.0],
            [0.5, 2.5, 1.5],
            [4.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    u = np.array(
        [
            [1.0, 0.1],
            [0.3, 0.8],
            [0.5, 0.4],
            [0.7, 0.2],
        ],
        dtype=np.float64,
    )
    v = np.array(
        [
            [0.2, 0.9],
            [0.4, 0.3],
            [0.8, 0.1],
        ],
        dtype=np.float64,
    )
    labels = np.array([1, 0, 1, 0, 1, 0, 1], dtype=np.int64)
    calls: dict[str, object] = {}

    model = SpectralCoclustering(n_clusters=3, random_state=0)

    def fake_svd(self: SpectralCoclustering, array: np.ndarray, n_sv: int, n_discard: int) -> tuple[np.ndarray, np.ndarray]:
        calls["n_sv"] = n_sv
        calls["n_discard"] = n_discard
        calls["array"] = np.asarray(array, dtype=np.float64)
        return u, v

    def fake_k_means(self: SpectralCoclustering, data: np.ndarray, n_clusters: int) -> tuple[np.ndarray, np.ndarray]:
        calls["embedding"] = np.asarray(data, dtype=np.float64)
        calls["kmeans_clusters"] = n_clusters
        return np.zeros((n_clusters, data.shape[1]), dtype=np.float64), labels

    model._svd = MethodType(fake_svd, model)
    model._k_means = MethodType(fake_k_means, model)
    model._fit(X)

    _, row_diag, col_diag = bicluster_scale_normalize(X)
    expected_embedding = cocluster_stacked_embedding(row_diag, u, col_diag, v)
    expected_row_labels, expected_column_labels = cocluster_split_labels(labels, X.shape[0])

    assert cocluster_singular_vector_count(model.n_clusters) == calls["n_sv"]
    assert calls["n_discard"] == 1
    assert np.allclose(calls["embedding"], expected_embedding)
    assert np.array_equal(model.row_labels_, expected_row_labels)
    assert np.array_equal(model.column_labels_, expected_column_labels)
    assert np.array_equal(model.rows_, cocluster_indicator_matrix(expected_row_labels, model.n_clusters))
    assert np.array_equal(model.columns_, cocluster_indicator_matrix(expected_column_labels, model.n_clusters))


@pytest.mark.parametrize(
    ("method", "n_components", "expected"),
    [
        ("bistochastic", 5, (6, 1)),
        ("scale", 4, (5, 1)),
        ("log", 3, (3, 0)),
    ],
)
def test_bicluster_effective_svd_dims_match_private_fit_arguments(
    method: str,
    n_components: int,
    expected: tuple[int, int],
) -> None:
    from sciona.atoms.ml.sklearn.cluster.bicluster_structure import bicluster_effective_svd_dims

    X = np.array(
        [
            [1.0, 2.0, 3.0],
            [2.0, 1.0, 0.5],
            [0.5, 1.5, 2.5],
            [3.5, 0.2, 1.0],
        ],
        dtype=np.float64,
    )
    calls: dict[str, int] = {}
    model = SpectralBiclustering(
        n_clusters=2,
        method=method,
        n_components=n_components,
        n_best=1,
        random_state=0,
    )

    def fake_svd(self: SpectralBiclustering, array: np.ndarray, n_sv: int, n_discard: int) -> tuple[np.ndarray, np.ndarray]:
        calls["n_sv"] = n_sv
        calls["n_discard"] = n_discard
        kept = n_sv - n_discard
        return np.ones((array.shape[0], kept), dtype=np.float64), np.ones((array.shape[1], kept), dtype=np.float64)

    def fake_fit_best_piecewise(self: SpectralBiclustering, vectors: np.ndarray, n_best: int, n_clusters: int) -> np.ndarray:
        return np.asarray(vectors[:n_best], dtype=np.float64)

    def fake_project_and_cluster(self: SpectralBiclustering, data: np.ndarray, vectors: np.ndarray, n_clusters: int) -> np.ndarray:
        return np.zeros(data.shape[0], dtype=np.int64)

    model._svd = MethodType(fake_svd, model)
    model._fit_best_piecewise = MethodType(fake_fit_best_piecewise, model)
    model._project_and_cluster = MethodType(fake_project_and_cluster, model)
    model._fit(X)

    assert bicluster_effective_svd_dims(method, n_components) == expected
    assert (calls["n_sv"], calls["n_discard"]) == expected


@pytest.mark.parametrize(
    ("n_clusters", "row_labels", "column_labels"),
    [
        (2, np.array([1, 0, 1, 0], dtype=np.int64), np.array([0, 1, 0], dtype=np.int64)),
        ((2, 3), np.array([1, 0, 1, 0], dtype=np.int64), np.array([2, 1, 0], dtype=np.int64)),
    ],
)
def test_bicluster_structure_helpers_match_private_fit_outputs(
    n_clusters: int | tuple[int, int],
    row_labels: np.ndarray,
    column_labels: np.ndarray,
) -> None:
    from sciona.atoms.ml.sklearn.cluster.bicluster_structure import (
        bicluster_indicator_grid,
        bicluster_resolve_cluster_counts,
    )

    X = np.array(
        [
            [2.0, 0.1, 1.0],
            [1.5, 2.2, 0.3],
            [0.4, 0.7, 3.1],
            [2.1, 1.1, 0.9],
        ],
        dtype=np.float64,
    )
    model = SpectralBiclustering(
        n_clusters=n_clusters,
        method="scale",
        n_components=3,
        n_best=1,
        random_state=0,
    )

    def fake_svd(self: SpectralBiclustering, array: np.ndarray, n_sv: int, n_discard: int) -> tuple[np.ndarray, np.ndarray]:
        kept = n_sv - n_discard
        return np.ones((array.shape[0], kept), dtype=np.float64), np.ones((array.shape[1], kept), dtype=np.float64)

    def fake_fit_best_piecewise(self: SpectralBiclustering, vectors: np.ndarray, n_best: int, n_clusters: int) -> np.ndarray:
        return np.asarray(vectors[:n_best], dtype=np.float64)

    def fake_project_and_cluster(self: SpectralBiclustering, data: np.ndarray, vectors: np.ndarray, n_clusters: int) -> np.ndarray:
        if data.shape[0] == X.shape[0]:
            return row_labels
        return column_labels

    model._svd = MethodType(fake_svd, model)
    model._fit_best_piecewise = MethodType(fake_fit_best_piecewise, model)
    model._project_and_cluster = MethodType(fake_project_and_cluster, model)
    model._fit(X)

    n_row_clusters, n_col_clusters = bicluster_resolve_cluster_counts(n_clusters)
    expected_rows, expected_columns = bicluster_indicator_grid(
        row_labels,
        column_labels,
        n_row_clusters,
        n_col_clusters,
    )

    assert np.array_equal(model.row_labels_, row_labels)
    assert np.array_equal(model.column_labels_, column_labels)
    assert np.array_equal(model.rows_, expected_rows)
    assert np.array_equal(model.columns_, expected_columns)


def test_bicluster_structure_contracts_reject_invalid_inputs() -> None:
    from sciona.atoms.ml.sklearn.cluster.bicluster_structure import (
        bicluster_effective_svd_dims,
        bicluster_indicator_grid,
        bicluster_resolve_cluster_counts,
        cocluster_indicator_matrix,
        cocluster_singular_vector_count,
        cocluster_split_labels,
        cocluster_stacked_embedding,
    )

    with pytest.raises(Exception):
        cocluster_singular_vector_count(0)
    with pytest.raises(Exception):
        cocluster_stacked_embedding(
            np.array([1.0, 2.0]),
            np.ones((3, 2), dtype=np.float64),
            np.array([1.0]),
            np.ones((1, 2), dtype=np.float64),
        )
    with pytest.raises(Exception):
        cocluster_split_labels(np.array([0, 1], dtype=np.int64), 2)
    with pytest.raises(Exception):
        cocluster_indicator_matrix(np.array([0, 1], dtype=np.int64), 0)
    with pytest.raises(Exception):
        bicluster_effective_svd_dims("bad", 2)
    with pytest.raises(Exception):
        bicluster_resolve_cluster_counts((2, 0))
    with pytest.raises(Exception):
        bicluster_indicator_grid(
            np.array([0, 1], dtype=np.int64),
            np.array([0, 1], dtype=np.int64),
            0,
            2,
        )
