from __future__ import annotations

import numpy as np
import pytest
from sklearn.manifold import SpectralEmbedding as SklearnSpectralEmbedding
from sklearn.manifold import spectral_embedding as sklearn_spectral_embedding
from sklearn.metrics.pairwise import rbf_kernel


def _data() -> np.ndarray:
    return np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, 1.0],
            [2.0, 2.0],
        ],
        dtype=np.float64,
    )


def test_spectral_embedding_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.manifold import SpectralEmbeddingState, spectral_embedding, spectral_embedding_fit

    assert SpectralEmbeddingState is not None
    assert callable(spectral_embedding)
    assert callable(spectral_embedding_fit)


def test_spectral_embedding_matches_sklearn_arpack_default() -> None:
    from sciona.atoms.ml.sklearn.manifold import spectral_embedding

    affinity = rbf_kernel(_data(), gamma=0.5)
    result = spectral_embedding(affinity, n_components=2, random_state=42)
    expected = sklearn_spectral_embedding(affinity, n_components=2, random_state=42)

    assert np.allclose(result, expected)


def test_spectral_embedding_keeps_first_vector_when_requested() -> None:
    from sciona.atoms.ml.sklearn.manifold import spectral_embedding

    affinity = rbf_kernel(_data(), gamma=0.5)
    result = spectral_embedding(affinity, n_components=2, random_state=42, drop_first=False)
    expected = sklearn_spectral_embedding(affinity, n_components=2, random_state=42, drop_first=False)

    assert np.allclose(result, expected)


def test_spectral_embedding_fit_matches_sklearn_rbf() -> None:
    from sciona.atoms.ml.sklearn.manifold import spectral_embedding_fit

    X = _data()
    state = spectral_embedding_fit(X, n_components=2, affinity="rbf", gamma=0.5, random_state=42)
    expected = SklearnSpectralEmbedding(n_components=2, affinity="rbf", gamma=0.5, random_state=42).fit(X)

    assert np.allclose(state.embedding, expected.embedding_)
    assert np.allclose(state.affinity_matrix, expected.affinity_matrix_)
    assert state.gamma == expected.gamma_


def test_spectral_embedding_fit_matches_sklearn_precomputed() -> None:
    from sciona.atoms.ml.sklearn.manifold import spectral_embedding_fit

    affinity = rbf_kernel(_data(), gamma=0.5)
    state = spectral_embedding_fit(affinity, n_components=2, affinity="precomputed", random_state=42)
    expected = SklearnSpectralEmbedding(n_components=2, affinity="precomputed", random_state=42).fit(affinity)

    assert np.allclose(state.embedding, expected.embedding_)
    assert np.allclose(state.affinity_matrix, expected.affinity_matrix_)


def test_spectral_embedding_rejects_out_of_scope_inputs() -> None:
    from sciona.atoms.ml.sklearn.manifold import spectral_embedding, spectral_embedding_fit

    X = _data()
    affinity = rbf_kernel(X, gamma=0.5)
    with pytest.raises(Exception):
        spectral_embedding(affinity, eigen_solver="lobpcg")
    with pytest.raises(Exception):
        spectral_embedding(affinity, n_components=affinity.shape[0])
    with pytest.raises(Exception):
        spectral_embedding_fit(X, affinity="nearest_neighbors")
    with pytest.raises(Exception):
        spectral_embedding_fit(X, affinity="rbf", n_neighbors=2)
    with pytest.raises(Exception):
        spectral_embedding_fit(X, affinity="precomputed")
