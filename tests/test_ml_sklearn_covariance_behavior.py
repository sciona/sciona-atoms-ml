from __future__ import annotations

import warnings

import numpy as np
from sklearn.covariance import (
    empirical_covariance as sklearn_empirical_covariance,
    shrunk_covariance as sklearn_shrunk_covariance,
)


def test_covariance_functions_import() -> None:
    from sciona.atoms.ml.sklearn.covariance import empirical_covariance, shrunk_covariance

    assert callable(empirical_covariance)
    assert callable(shrunk_covariance)


def test_empirical_covariance_matches_sklearn_centered_modes() -> None:
    from sciona.atoms.ml.sklearn.covariance import empirical_covariance

    X = np.array(
        [
            [0.0, 1.0, 2.0],
            [1.0, 2.0, 4.0],
            [2.0, 4.0, 8.0],
            [3.0, 8.0, 16.0],
        ],
        dtype=np.float64,
    )

    assert np.allclose(empirical_covariance(X), sklearn_empirical_covariance(X))
    assert np.allclose(
        empirical_covariance(X, assume_centered=True),
        sklearn_empirical_covariance(X, assume_centered=True),
    )


def test_empirical_covariance_handles_1d_like_sklearn() -> None:
    from sciona.atoms.ml.sklearn.covariance import empirical_covariance

    X = np.array([1.0, 2.0, 3.0], dtype=np.float64)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        result = empirical_covariance(X)
        expected = sklearn_empirical_covariance(X)
    assert np.allclose(result, expected)
    assert result.shape == (3, 3)


def test_shrunk_covariance_matches_sklearn_single_and_batched() -> None:
    from sciona.atoms.ml.sklearn.covariance import shrunk_covariance

    cov = np.array([[2.0, 0.5], [0.5, 1.0]], dtype=np.float64)
    stacked = np.stack([cov, cov * 2.0])

    assert np.allclose(shrunk_covariance(cov, shrinkage=0.25), sklearn_shrunk_covariance(cov, shrinkage=0.25))
    assert np.allclose(
        shrunk_covariance(stacked, shrinkage=0.5),
        sklearn_shrunk_covariance(stacked, shrinkage=0.5),
    )


def test_covariance_contract_shapes_and_symmetry() -> None:
    from sciona.atoms.ml.sklearn.covariance import empirical_covariance, shrunk_covariance

    X = np.arange(12, dtype=np.float64).reshape(4, 3)
    cov = empirical_covariance(X)
    shrunk = shrunk_covariance(cov)

    assert cov.shape == (3, 3)
    assert np.allclose(cov, cov.T)
    assert shrunk.shape == cov.shape
    assert np.isfinite(shrunk).all()
