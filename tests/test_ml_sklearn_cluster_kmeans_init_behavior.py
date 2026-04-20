from __future__ import annotations

import numpy as np
from sklearn.cluster import kmeans_plusplus as sklearn_kmeans_plusplus
from sklearn.utils.extmath import row_norms


def test_kmeans_init_atom_import() -> None:
    from sciona.atoms.ml.sklearn.cluster import kmeans_plusplus

    assert callable(kmeans_plusplus)


def test_kmeans_plusplus_matches_sklearn() -> None:
    from sciona.atoms.ml.sklearn.cluster import kmeans_plusplus

    X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]], dtype=np.float64)
    centers, indices = kmeans_plusplus(X, 2, random_state=0)
    expected_centers, expected_indices = sklearn_kmeans_plusplus(X, 2, random_state=0)

    assert np.allclose(centers, expected_centers)
    assert np.array_equal(indices, expected_indices)


def test_kmeans_plusplus_matches_sklearn_with_weights_and_norms() -> None:
    from sciona.atoms.ml.sklearn.cluster import kmeans_plusplus

    X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]], dtype=np.float64)
    sample_weight = np.array([1, 2, 1, 1, 1, 1], dtype=np.float64)
    norms = row_norms(X, squared=True)

    centers, indices = kmeans_plusplus(
        X,
        3,
        sample_weight=sample_weight,
        x_squared_norms=norms,
        random_state=42,
        n_local_trials=2,
    )
    expected_centers, expected_indices = sklearn_kmeans_plusplus(
        X,
        3,
        sample_weight=sample_weight,
        x_squared_norms=norms,
        random_state=42,
        n_local_trials=2,
    )

    assert np.allclose(centers, expected_centers)
    assert np.array_equal(indices, expected_indices)


def test_kmeans_plusplus_rejects_too_many_clusters() -> None:
    from sciona.atoms.ml.sklearn.cluster import kmeans_plusplus

    X = np.array([[0, 0], [1, 1]], dtype=np.float64)
    try:
        kmeans_plusplus(X, 3, random_state=0)
    except Exception as exc:
        assert "n_clusters" in str(exc)
    else:
        raise AssertionError("expected n_clusters validation failure")
