from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.utils.extmath import softmax


def test_nca_helper_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.neighbors.nca import (
        nca_linear_transform,
        nca_loss_gradient,
        nca_neighbor_probabilities,
        nca_same_class_mask,
    )

    assert callable(nca_same_class_mask)
    assert callable(nca_linear_transform)
    assert callable(nca_neighbor_probabilities)
    assert callable(nca_loss_gradient)


def test_nca_same_class_mask_matches_sklearn_fit_mask_construction() -> None:
    from sciona.atoms.ml.sklearn.neighbors.nca import nca_same_class_mask

    y = np.array([0, 1, 0, 2, 1], dtype=np.int64)
    expected = y[:, np.newaxis] == y[np.newaxis, :]

    assert np.array_equal(nca_same_class_mask(y), expected)


def test_nca_linear_transform_matches_sklearn_transform() -> None:
    from sciona.atoms.ml.sklearn.neighbors.nca import nca_linear_transform

    X = np.array([[1.0, 2.0, -1.0], [0.5, -3.0, 4.0], [2.0, 1.0, 0.0]], dtype=np.float64)
    components = np.array([[0.5, -1.0, 0.25], [1.5, 0.0, -0.5]], dtype=np.float64)

    nca = NeighborhoodComponentsAnalysis()
    nca.components_ = components
    nca.n_features_in_ = X.shape[1]

    assert np.allclose(nca_linear_transform(X, components), nca.transform(X))


def test_nca_neighbor_probabilities_match_source_steps() -> None:
    from sciona.atoms.ml.sklearn.neighbors.nca import nca_neighbor_probabilities

    X_embedded = np.array([[0.0, 0.0], [1.0, -1.0], [2.0, 0.5], [-0.5, 2.0]], dtype=np.float64)
    distances = pairwise_distances(X_embedded, squared=True)
    np.fill_diagonal(distances, np.inf)
    expected = softmax(-distances)

    result = nca_neighbor_probabilities(X_embedded)
    assert np.allclose(result, expected)
    assert np.allclose(result.sum(axis=1), 1.0)
    assert np.allclose(np.diag(result), 0.0)


def test_nca_loss_gradient_matches_sklearn_private_helper_for_both_signs() -> None:
    from sciona.atoms.ml.sklearn.neighbors.nca import nca_loss_gradient, nca_same_class_mask

    X = np.array(
        [
            [0.0, 1.0, 2.0],
            [1.0, 0.5, -1.0],
            [2.0, -0.5, 0.25],
            [-1.0, 1.5, 1.0],
        ],
        dtype=np.float64,
    )
    y = np.array([0, 1, 0, 1], dtype=np.int64)
    components = np.array([[0.5, -0.2, 0.1], [1.0, 0.25, -0.75]], dtype=np.float64)
    transformation = components.ravel()
    same_class_mask = nca_same_class_mask(y)

    nca = NeighborhoodComponentsAnalysis()
    nca.n_iter_ = 1
    nca.verbose = 0

    for sign in (1.0, -1.0):
        expected_loss, expected_gradient = nca._loss_grad_lbfgs(transformation, X, same_class_mask, sign=sign)
        result_loss, result_gradient = nca_loss_gradient(transformation, X, same_class_mask, sign=sign)
        assert result_loss == pytest.approx(expected_loss)
        assert np.allclose(result_gradient, expected_gradient)


def test_contracts_reject_invalid_nca_inputs() -> None:
    from sciona.atoms.ml.sklearn.neighbors.nca import (
        nca_linear_transform,
        nca_loss_gradient,
        nca_neighbor_probabilities,
        nca_same_class_mask,
    )

    with pytest.raises(ViolationError):
        nca_same_class_mask(np.array([0.0, 1.0], dtype=np.float64))

    with pytest.raises(ViolationError):
        nca_linear_transform(np.ones((3, 2), dtype=np.float64), np.ones((2, 3), dtype=np.float64))

    with pytest.raises(ViolationError):
        nca_neighbor_probabilities(np.ones((1, 2), dtype=np.float64))

    with pytest.raises(ViolationError):
        nca_loss_gradient(
            np.ones(5, dtype=np.float64),
            np.ones((3, 2), dtype=np.float64),
            np.ones((3, 3), dtype=np.bool_),
        )
