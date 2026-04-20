from __future__ import annotations

import numpy as np
import pytest
from sklearn.manifold import ClassicalMDS as SklearnClassicalMDS
from sklearn.metrics import pairwise_distances


def _data() -> np.ndarray:
    return np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, 0.0],
        ],
        dtype=np.float64,
    )


def test_classical_mds_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.manifold import (
        ClassicalMDSState,
        classical_mds_dissimilarity_matrix,
        classical_mds_double_center,
        classical_mds_fit,
    )

    assert ClassicalMDSState is not None
    assert callable(classical_mds_dissimilarity_matrix)
    assert callable(classical_mds_double_center)
    assert callable(classical_mds_fit)


def test_classical_mds_dissimilarity_matches_sklearn_pairwise() -> None:
    from sciona.atoms.ml.sklearn.manifold import classical_mds_dissimilarity_matrix

    X = _data()
    result = classical_mds_dissimilarity_matrix(X)

    assert np.allclose(result, pairwise_distances(X, metric="euclidean"))
    assert np.allclose(result, result.T)


def test_classical_mds_double_center_matches_sklearn_formula() -> None:
    from sciona.atoms.ml.sklearn.manifold import classical_mds_double_center

    D = pairwise_distances(_data())
    expected = D**2
    expected = expected.astype(np.float64)
    expected -= np.mean(expected, axis=0)
    expected -= np.mean(expected, axis=1, keepdims=True)
    expected *= -0.5

    result = classical_mds_double_center(D)

    assert np.allclose(result, expected)
    assert np.allclose(np.mean(result, axis=0), 0.0)
    assert np.allclose(np.mean(result, axis=1), 0.0)


def test_classical_mds_fit_matches_sklearn_euclidean() -> None:
    from sciona.atoms.ml.sklearn.manifold import classical_mds_fit

    X = _data()
    state = classical_mds_fit(X, n_components=2)
    expected = SklearnClassicalMDS(n_components=2).fit(X)

    assert state.n_components == 2
    assert state.metric == "euclidean"
    assert state.n_features_in == X.shape[1]
    assert np.allclose(state.embedding, expected.embedding_)
    assert np.allclose(state.dissimilarity_matrix, expected.dissimilarity_matrix_)
    assert np.allclose(state.eigenvalues, expected.eigenvalues_)


def test_classical_mds_fit_matches_sklearn_precomputed() -> None:
    from sciona.atoms.ml.sklearn.manifold import classical_mds_fit

    X = _data()
    D = pairwise_distances(X)
    state = classical_mds_fit(D, n_components=2, metric="precomputed")
    expected = SklearnClassicalMDS(n_components=2, metric="precomputed").fit(D)

    assert state.metric == "precomputed"
    assert state.n_features_in == D.shape[1]
    assert np.allclose(state.embedding, expected.embedding_)
    assert np.allclose(state.dissimilarity_matrix, expected.dissimilarity_matrix_)
    assert np.allclose(state.eigenvalues, expected.eigenvalues_)


def test_classical_mds_rejects_invalid_or_unsupported_inputs() -> None:
    from sciona.atoms.ml.sklearn.manifold import classical_mds_dissimilarity_matrix, classical_mds_fit

    X = _data()
    with pytest.raises(Exception):
        classical_mds_fit(X, n_components=0)
    with pytest.raises(Exception):
        classical_mds_fit(X, metric="manhattan")
    with pytest.raises(Exception):
        classical_mds_fit(X, metric_params={})
    with pytest.raises(Exception):
        classical_mds_fit(X[:, :1], metric="precomputed")
    with pytest.raises(Exception):
        classical_mds_dissimilarity_matrix(np.array([[0.0, 1.0], [2.0, 0.0]]), metric="precomputed")
