from __future__ import annotations

import numpy as np
import pytest


def _dictionary() -> np.ndarray:
    return np.array(
        [
            [1.0, 0.0, 0.5],
            [0.0, 1.0, -0.5],
            [1.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )


def _data() -> np.ndarray:
    return np.array(
        [
            [1.0, 0.0, 1.0],
            [0.0, 1.0, -1.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )


def test_sparse_encode_precompute_dispatch_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.decomposition.sparse_encode_precompute_dispatch import (
        sparse_encode_dispatched_covariance,
        sparse_encode_dispatched_gram,
        sparse_encode_resolved_copy_cov,
    )

    assert callable(sparse_encode_dispatched_gram)
    assert callable(sparse_encode_dispatched_covariance)
    assert callable(sparse_encode_resolved_copy_cov)


def test_sparse_encode_precompute_dispatch_matches_source_logic() -> None:
    from sciona.atoms.ml.sklearn.decomposition.sparse_encode_precompute_dispatch import (
        sparse_encode_dispatched_covariance,
        sparse_encode_dispatched_gram,
        sparse_encode_resolved_copy_cov,
    )

    X = _data()
    dictionary = _dictionary()

    assert np.array_equal(
        sparse_encode_dispatched_gram(None, dictionary, "omp"),
        dictionary @ dictionary.T,
    )
    assert sparse_encode_dispatched_gram(None, dictionary, "threshold") is None

    provided_gram = np.eye(dictionary.shape[0], dtype=np.float64)
    assert np.array_equal(
        sparse_encode_dispatched_gram(provided_gram, dictionary, "lasso_cd"),
        provided_gram,
    )

    assert np.array_equal(
        sparse_encode_dispatched_covariance(None, X, dictionary, "omp"),
        dictionary @ X.T,
    )
    assert sparse_encode_dispatched_covariance(None, X, dictionary, "lasso_cd") is None

    provided_cov = np.ones((dictionary.shape[0], X.shape[0]), dtype=np.float64)
    assert np.array_equal(
        sparse_encode_dispatched_covariance(provided_cov, X, dictionary, "lasso_cd"),
        provided_cov,
    )

    assert sparse_encode_resolved_copy_cov(True, None, "omp") is False
    assert sparse_encode_resolved_copy_cov(True, None, "lasso_cd") is True
    assert sparse_encode_resolved_copy_cov(False, provided_cov, "lasso_lars") is False


def test_sparse_encode_precompute_dispatch_contracts() -> None:
    from sciona.atoms.ml.sklearn.decomposition.sparse_encode_precompute_dispatch import (
        sparse_encode_dispatched_covariance,
        sparse_encode_dispatched_gram,
        sparse_encode_resolved_copy_cov,
    )

    X = _data()
    dictionary = _dictionary()

    with pytest.raises(Exception):
        sparse_encode_dispatched_gram(np.array([[1.0, np.nan]], dtype=np.float64), dictionary, "omp")

    with pytest.raises(Exception):
        sparse_encode_dispatched_covariance(np.ones((2, 2), dtype=np.float64), X, dictionary, "omp")

    with pytest.raises(Exception):
        sparse_encode_resolved_copy_cov("yes", None, "omp")  # type: ignore[arg-type]
