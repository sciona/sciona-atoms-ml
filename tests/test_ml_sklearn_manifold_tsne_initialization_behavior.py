from __future__ import annotations

from types import MethodType

import numpy as np
import pytest
from icontract import ViolationError
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def _dataset() -> np.ndarray:
    return np.array(
        [
            [0.0, 1.0, 2.0],
            [1.0, 0.0, 1.5],
            [2.0, 1.0, 0.0],
            [3.0, 2.0, 1.0],
            [4.0, 1.5, 0.5],
            [5.0, 2.5, 1.5],
        ],
        dtype=np.float64,
    )


def test_tsne_initialization_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.manifold.tsne_initialization import (
        tsne_auto_learning_rate,
        tsne_barnes_hut_neighbor_count,
        tsne_degrees_of_freedom,
        tsne_pca_rescale_embedding,
        tsne_random_initialize_embedding,
    )

    assert callable(tsne_auto_learning_rate)
    assert callable(tsne_barnes_hut_neighbor_count)
    assert callable(tsne_degrees_of_freedom)
    assert callable(tsne_pca_rescale_embedding)
    assert callable(tsne_random_initialize_embedding)


def test_tsne_auto_learning_rate_matches_private_fit_assignment() -> None:
    from sciona.atoms.ml.sklearn.manifold.tsne_initialization import tsne_auto_learning_rate

    X = _dataset()
    tsne = TSNE(
        n_components=2,
        init="random",
        learning_rate="auto",
        perplexity=2.0,
        random_state=0,
        method="exact",
    )

    def fake_tsne(self, P, degrees_of_freedom, n_samples, *, X_embedded, neighbors, skip_num_points):
        return X_embedded

    tsne._tsne = MethodType(fake_tsne, tsne)
    tsne._fit(X)

    actual = tsne_auto_learning_rate(X.shape[0], tsne.early_exaggeration)

    assert actual == pytest.approx(tsne.learning_rate_)


def test_tsne_barnes_hut_neighbor_count_matches_formula() -> None:
    from sciona.atoms.ml.sklearn.manifold.tsne_initialization import tsne_barnes_hut_neighbor_count

    assert tsne_barnes_hut_neighbor_count(10, 30.0) == 9
    assert tsne_barnes_hut_neighbor_count(200, 30.0) == 91


def test_tsne_random_initialize_embedding_matches_private_fit_random_init() -> None:
    from sciona.atoms.ml.sklearn.manifold.tsne_initialization import tsne_random_initialize_embedding

    X = _dataset()
    seed = 11
    tsne = TSNE(
        n_components=2,
        init="random",
        learning_rate=200.0,
        perplexity=2.0,
        random_state=seed,
        method="exact",
    )

    def fake_tsne(self, P, degrees_of_freedom, n_samples, *, X_embedded, neighbors, skip_num_points):
        return X_embedded

    tsne._tsne = MethodType(fake_tsne, tsne)
    expected = tsne._fit(X)
    actual = tsne_random_initialize_embedding(X.shape[0], 2, random_state=seed)

    assert expected.dtype == np.float32
    assert np.array_equal(actual, expected)


def test_tsne_pca_rescale_embedding_matches_private_fit_pca_init() -> None:
    from sciona.atoms.ml.sklearn.manifold.tsne_initialization import tsne_pca_rescale_embedding

    X = _dataset()
    seed = 7
    tsne = TSNE(
        n_components=2,
        init="pca",
        learning_rate=200.0,
        perplexity=2.0,
        random_state=seed,
        method="exact",
    )

    def fake_tsne(self, P, degrees_of_freedom, n_samples, *, X_embedded, neighbors, skip_num_points):
        return X_embedded

    tsne._tsne = MethodType(fake_tsne, tsne)
    expected = tsne._fit(X)

    pca = PCA(n_components=2, svd_solver="randomized", random_state=seed)
    pca.set_output(transform="default")
    raw_embedding = pca.fit_transform(X).astype(np.float32, copy=False)
    actual = tsne_pca_rescale_embedding(raw_embedding)

    assert expected.dtype == np.float32
    assert np.allclose(actual, expected)


def test_tsne_degrees_of_freedom_matches_private_fit_argument() -> None:
    from sciona.atoms.ml.sklearn.manifold.tsne_initialization import tsne_degrees_of_freedom

    X = _dataset()
    captured: dict[str, int] = {}
    tsne = TSNE(
        n_components=3,
        init="random",
        learning_rate=200.0,
        perplexity=2.0,
        random_state=0,
        method="exact",
    )

    def fake_tsne(self, P, degrees_of_freedom, n_samples, *, X_embedded, neighbors, skip_num_points):
        captured["degrees_of_freedom"] = degrees_of_freedom
        return X_embedded

    tsne._tsne = MethodType(fake_tsne, tsne)
    tsne._fit(X)

    assert tsne_degrees_of_freedom(3) == captured["degrees_of_freedom"]


def test_tsne_initialization_contracts_reject_invalid_inputs() -> None:
    from sciona.atoms.ml.sklearn.manifold.tsne_initialization import (
        tsne_auto_learning_rate,
        tsne_barnes_hut_neighbor_count,
        tsne_degrees_of_freedom,
        tsne_pca_rescale_embedding,
        tsne_random_initialize_embedding,
    )

    with pytest.raises(ViolationError):
        tsne_auto_learning_rate(1, 12.0)

    with pytest.raises(ViolationError):
        tsne_barnes_hut_neighbor_count(5, 0.0)

    with pytest.raises(ViolationError):
        tsne_random_initialize_embedding(0, 2)

    with pytest.raises(ViolationError):
        tsne_pca_rescale_embedding(np.zeros((4, 2), dtype=np.float32))

    with pytest.raises(ViolationError):
        tsne_degrees_of_freedom(0)
