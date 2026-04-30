from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError
from scipy.sparse import csr_matrix
from sklearn.manifold import _t_sne as sklearn_tsne
from sklearn.neighbors import NearestNeighbors


def _neighbor_distances() -> csr_matrix:
    X = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [0.5, 0.2],
        ],
        dtype=np.float64,
    )
    nn = NearestNeighbors(n_neighbors=3, metric="euclidean")
    nn.fit(X)
    distances = nn.kneighbors_graph(mode="distance").tocsr()
    distances.sort_indices()
    return distances


def test_tsne_nn_probabilities_imports() -> None:
    from sciona.atoms.ml.sklearn.manifold.tsne_nn_probabilities import (
        tsne_nn_conditional_probability_matrix,
        tsne_nn_distance_blocks,
        tsne_nn_joint_probabilities,
    )

    assert callable(tsne_nn_distance_blocks)
    assert callable(tsne_nn_conditional_probability_matrix)
    assert callable(tsne_nn_joint_probabilities)


def test_tsne_nn_distance_blocks_match_sklearn_preprocessing() -> None:
    from sciona.atoms.ml.sklearn.manifold.tsne_nn_probabilities import tsne_nn_distance_blocks

    distances = _neighbor_distances()
    expected = distances.data.reshape(distances.shape[0], -1).astype(np.float32, copy=False)

    actual = tsne_nn_distance_blocks(distances)
    assert actual.dtype == np.float32
    assert np.array_equal(actual, expected)


def test_tsne_nn_joint_probabilities_match_sklearn_private_helper() -> None:
    from sciona.atoms.ml.sklearn.manifold.tsne_nn_probabilities import (
        tsne_nn_conditional_probability_matrix,
        tsne_nn_distance_blocks,
        tsne_nn_joint_probabilities,
    )

    distances = _neighbor_distances()
    blocks = tsne_nn_distance_blocks(distances)
    conditional = sklearn_tsne._utils._binary_search_perplexity(blocks, 2.0, 0)

    conditional_matrix = tsne_nn_conditional_probability_matrix(
        conditional,
        distances.indices.astype(np.int64, copy=False),
        distances.indptr.astype(np.int64, copy=False),
        n_samples=distances.shape[0],
    )
    actual = tsne_nn_joint_probabilities(conditional_matrix)
    expected = sklearn_tsne._joint_probabilities_nn(distances.copy(), 2.0, 0)

    assert isinstance(actual, csr_matrix)
    assert actual.shape == expected.shape
    assert np.allclose(actual.toarray(), expected.toarray())
    assert np.isclose(actual.sum(), 1.0)


def test_tsne_nn_probabilities_contracts_reject_invalid_inputs() -> None:
    from sciona.atoms.ml.sklearn.manifold.tsne_nn_probabilities import (
        tsne_nn_conditional_probability_matrix,
        tsne_nn_distance_blocks,
        tsne_nn_joint_probabilities,
    )

    with pytest.raises(ViolationError):
        tsne_nn_distance_blocks(np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64))  # type: ignore[arg-type]

    with pytest.raises(ViolationError):
        tsne_nn_conditional_probability_matrix(
            np.ones((2, 2), dtype=np.float64),
            np.array([0, 1, 0], dtype=np.int64),
            np.array([0, 2, 4], dtype=np.int64),
            n_samples=2,
        )

    with pytest.raises(ViolationError):
        tsne_nn_joint_probabilities(
            csr_matrix(np.array([[0.0, -1.0], [0.0, 0.0]], dtype=np.float64))
        )
