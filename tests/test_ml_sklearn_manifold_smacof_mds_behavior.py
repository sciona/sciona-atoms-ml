from __future__ import annotations

import numpy as np
import pytest
from sklearn.manifold import MDS as SklearnMDS
from sklearn.manifold import smacof as sklearn_smacof
from sklearn.metrics import pairwise_distances


def _data() -> np.ndarray:
    return np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, 1.0],
        ],
        dtype=np.float64,
    )


def test_smacof_mds_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.manifold import MDSState, SMACOFState, mds_fit, smacof

    assert MDSState is not None
    assert SMACOFState is not None
    assert callable(smacof)
    assert callable(mds_fit)


def test_smacof_matches_sklearn_metric_single_init() -> None:
    from sciona.atoms.ml.sklearn.manifold import smacof

    dissimilarities = pairwise_distances(_data())
    state = smacof(dissimilarities, n_components=2, n_init=1, random_state=42, max_iter=20, eps=1e-6)
    embedding, stress, n_iter = sklearn_smacof(
        dissimilarities,
        n_components=2,
        n_init=1,
        random_state=42,
        max_iter=20,
        eps=1e-6,
        return_n_iter=True,
    )

    assert np.allclose(state.embedding, embedding)
    assert np.isclose(state.stress, stress)
    assert state.n_iter == n_iter


def test_smacof_normalized_stress_matches_sklearn() -> None:
    from sciona.atoms.ml.sklearn.manifold import smacof

    dissimilarities = pairwise_distances(_data())
    state = smacof(dissimilarities, n_components=2, n_init=1, random_state=7, max_iter=15, normalized_stress=True)
    embedding, stress, n_iter = sklearn_smacof(
        dissimilarities,
        n_components=2,
        n_init=1,
        random_state=7,
        max_iter=15,
        return_n_iter=True,
        normalized_stress=True,
    )

    assert np.allclose(state.embedding, embedding)
    assert np.isclose(state.stress, stress)
    assert state.n_iter == n_iter


def test_mds_fit_matches_sklearn_euclidean() -> None:
    from sciona.atoms.ml.sklearn.manifold import mds_fit

    X = _data()
    state = mds_fit(X, n_components=2, n_init=1, random_state=42, max_iter=20, eps=1e-6)
    expected = SklearnMDS(
        n_components=2,
        n_init=1,
        init="random",
        random_state=42,
        max_iter=20,
        eps=1e-6,
        metric="euclidean",
    ).fit(X)

    assert np.allclose(state.embedding, expected.embedding_)
    assert np.isclose(state.stress, expected.stress_)
    assert state.n_iter == expected.n_iter_
    assert np.allclose(state.dissimilarity_matrix, expected.dissimilarity_matrix_)


def test_mds_fit_matches_sklearn_precomputed() -> None:
    from sciona.atoms.ml.sklearn.manifold import mds_fit

    dissimilarities = pairwise_distances(_data())
    state = mds_fit(dissimilarities, n_components=2, n_init=1, random_state=42, max_iter=20, metric="precomputed")
    expected = SklearnMDS(n_components=2, n_init=1, init="random", random_state=42, max_iter=20, metric="precomputed").fit(dissimilarities)

    assert np.allclose(state.embedding, expected.embedding_)
    assert np.isclose(state.stress, expected.stress_)
    assert state.n_iter == expected.n_iter_


def test_smacof_mds_rejects_out_of_scope_inputs() -> None:
    from sciona.atoms.ml.sklearn.manifold import mds_fit, smacof

    dissimilarities = pairwise_distances(_data())
    with pytest.raises(Exception):
        smacof(dissimilarities, metric=False)
    with pytest.raises(Exception):
        smacof(dissimilarities, n_init=2)
    with pytest.raises(Exception):
        smacof(dissimilarities, n_jobs=2)
    with pytest.raises(Exception):
        mds_fit(_data(), metric_mds=False)
    with pytest.raises(Exception):
        mds_fit(_data(), n_init=2)
    with pytest.raises(Exception):
        mds_fit(_data(), metric="manhattan")
    with pytest.raises(Exception):
        mds_fit(_data(), metric="precomputed")
