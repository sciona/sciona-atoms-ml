from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError
from sklearn.decomposition import sparse_encode as sklearn_sparse_encode


def _data() -> tuple[np.ndarray, np.ndarray]:
    X = np.array(
        [
            [-1.0, -1.0, -1.0],
            [0.0, 0.0, 3.0],
            [1.0, 2.0, -0.5],
        ],
        dtype=np.float64,
    )
    dictionary = np.array(
        [
            [0.0, 1.0, 0.0],
            [-1.0, -1.0, 2.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    return X, dictionary


def test_sparse_encode_preprocessing_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.decomposition.sparse_encode_preprocessing import (
        sparse_encode_covariance,
        sparse_encode_gram,
        sparse_encode_regularization,
        sparse_encode_threshold,
    )

    assert callable(sparse_encode_regularization)
    assert callable(sparse_encode_gram)
    assert callable(sparse_encode_covariance)
    assert callable(sparse_encode_threshold)


def test_sparse_encode_regularization_matches_sklearn_default_rules() -> None:
    from sciona.atoms.ml.sklearn.decomposition.sparse_encode_preprocessing import sparse_encode_regularization

    assert sparse_encode_regularization("lasso_lars", n_features=20, n_components=7, alpha=None) == pytest.approx(1.0)
    assert sparse_encode_regularization("threshold", n_features=20, n_components=7, alpha=0.3) == pytest.approx(0.3)
    assert sparse_encode_regularization("lars", n_features=25, n_components=4, n_nonzero_coefs=None) == pytest.approx(2.5)
    assert sparse_encode_regularization("omp", n_features=3, n_components=10, n_nonzero_coefs=None) == pytest.approx(1.0)
    assert sparse_encode_regularization("omp", n_features=12, n_components=8, n_nonzero_coefs=5) == pytest.approx(5.0)


def test_sparse_encode_gram_and_covariance_match_precomputes() -> None:
    from sciona.atoms.ml.sklearn.decomposition.sparse_encode_preprocessing import (
        sparse_encode_covariance,
        sparse_encode_gram,
    )

    X, dictionary = _data()

    assert np.allclose(sparse_encode_gram(dictionary), dictionary @ dictionary.T)
    assert np.allclose(sparse_encode_covariance(X, dictionary), dictionary @ X.T)


def test_sparse_encode_threshold_matches_sklearn_public_helper() -> None:
    from sciona.atoms.ml.sklearn.decomposition.sparse_encode_preprocessing import sparse_encode_threshold

    X, dictionary = _data()
    result = sparse_encode_threshold(X, dictionary, alpha=0.75)
    expected = sklearn_sparse_encode(X, dictionary, algorithm="threshold", alpha=0.75)

    assert np.allclose(result, expected)


def test_sparse_encode_threshold_matches_sklearn_with_precomputed_covariance() -> None:
    from sciona.atoms.ml.sklearn.decomposition.sparse_encode_preprocessing import (
        sparse_encode_covariance,
        sparse_encode_threshold,
    )

    X, dictionary = _data()
    cov = sparse_encode_covariance(X, dictionary)

    assert np.allclose(
        sparse_encode_threshold(X, dictionary, cov=cov, alpha=0.5),
        sklearn_sparse_encode(X, dictionary, algorithm="threshold", alpha=0.5, cov=cov),
    )


def test_sparse_encode_threshold_positive_matches_sklearn_public_helper() -> None:
    from sciona.atoms.ml.sklearn.decomposition.sparse_encode_preprocessing import sparse_encode_threshold

    X, dictionary = _data()
    result = sparse_encode_threshold(X, dictionary, alpha=0.5, positive=True)
    expected = sklearn_sparse_encode(X, dictionary, algorithm="threshold", alpha=0.5, positive=True)

    assert np.allclose(result, expected)
    assert np.all(result >= 0.0)


def test_sparse_encode_threshold_uses_default_alpha_like_sklearn() -> None:
    from sciona.atoms.ml.sklearn.decomposition.sparse_encode_preprocessing import sparse_encode_threshold

    X, dictionary = _data()

    assert np.allclose(
        sparse_encode_threshold(X, dictionary, alpha=None),
        sklearn_sparse_encode(X, dictionary, algorithm="threshold", alpha=None),
    )


def test_contracts_reject_invalid_sparse_encode_preprocessing_inputs() -> None:
    from sciona.atoms.ml.sklearn.decomposition.sparse_encode_preprocessing import (
        sparse_encode_covariance,
        sparse_encode_gram,
        sparse_encode_regularization,
        sparse_encode_threshold,
    )

    X, dictionary = _data()

    with pytest.raises(ViolationError):
        sparse_encode_regularization("bad", n_features=3, n_components=2)

    with pytest.raises(ViolationError):
        sparse_encode_regularization("omp", n_features=3, n_components=2, n_nonzero_coefs=0)

    with pytest.raises(ViolationError):
        sparse_encode_gram(np.array([[1.0, np.nan]], dtype=np.float64))

    with pytest.raises(ViolationError):
        sparse_encode_covariance(X[:, :2], dictionary)

    with pytest.raises(ViolationError):
        sparse_encode_threshold(X, dictionary, cov=np.ones((2, 2), dtype=np.float64))

    with pytest.raises(ViolationError):
        sparse_encode_threshold(X, dictionary, alpha=-0.1)
