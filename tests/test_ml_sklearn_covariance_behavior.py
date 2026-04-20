from __future__ import annotations

import warnings

import numpy as np
from sklearn.covariance import (
    empirical_covariance as sklearn_empirical_covariance,
    ledoit_wolf as sklearn_ledoit_wolf,
    ledoit_wolf_shrinkage as sklearn_ledoit_wolf_shrinkage,
    oas as sklearn_oas,
    shrunk_covariance as sklearn_shrunk_covariance,
)


def test_covariance_functions_import() -> None:
    from sciona.atoms.ml.sklearn.covariance import (
        empirical_covariance,
        ledoit_wolf,
        ledoit_wolf_shrinkage,
        oas,
        shrunk_covariance,
    )

    assert callable(empirical_covariance)
    assert callable(ledoit_wolf)
    assert callable(ledoit_wolf_shrinkage)
    assert callable(oas)
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


def test_ledoit_wolf_shrinkage_matches_sklearn_centered_and_blocked() -> None:
    from sciona.atoms.ml.sklearn.covariance import ledoit_wolf_shrinkage

    X = np.array(
        [
            [0.0, 1.0, 3.0],
            [1.0, 2.0, 4.0],
            [2.0, 4.0, 7.0],
            [4.0, 8.0, 9.0],
            [5.0, 9.0, 12.0],
        ],
        dtype=np.float64,
    )

    assert np.allclose(ledoit_wolf_shrinkage(X, block_size=2), sklearn_ledoit_wolf_shrinkage(X, block_size=2))
    assert np.allclose(
        ledoit_wolf_shrinkage(X, assume_centered=True, block_size=2),
        sklearn_ledoit_wolf_shrinkage(X, assume_centered=True, block_size=2),
    )


def test_ledoit_wolf_matches_sklearn_covariance_and_shrinkage() -> None:
    from sciona.atoms.ml.sklearn.covariance import ledoit_wolf

    X = np.array(
        [
            [0.0, 1.0, 3.0],
            [1.0, 2.0, 4.0],
            [2.0, 4.0, 7.0],
            [4.0, 8.0, 9.0],
            [5.0, 9.0, 12.0],
        ],
        dtype=np.float64,
    )

    cov, shrinkage = ledoit_wolf(X, block_size=2)
    expected_cov, expected_shrinkage = sklearn_ledoit_wolf(X, block_size=2)
    assert np.allclose(cov, expected_cov)
    assert np.allclose(shrinkage, expected_shrinkage)

    single_feature = X[:, :1]
    cov_one, shrinkage_one = ledoit_wolf(single_feature)
    expected_cov_one, expected_shrinkage_one = sklearn_ledoit_wolf(single_feature)
    assert np.allclose(cov_one, expected_cov_one)
    assert shrinkage_one == expected_shrinkage_one == 0.0


def test_oas_matches_sklearn_covariance_and_shrinkage() -> None:
    from sciona.atoms.ml.sklearn.covariance import oas

    X = np.array(
        [
            [0.0, 1.0, 3.0],
            [1.0, 2.0, 4.0],
            [2.0, 4.0, 7.0],
            [4.0, 8.0, 9.0],
            [5.0, 9.0, 12.0],
        ],
        dtype=np.float64,
    )

    cov, shrinkage = oas(X)
    expected_cov, expected_shrinkage = sklearn_oas(X)
    assert np.allclose(cov, expected_cov)
    assert np.allclose(shrinkage, expected_shrinkage)

    centered_cov, centered_shrinkage = oas(X, assume_centered=True)
    expected_centered_cov, expected_centered_shrinkage = sklearn_oas(X, assume_centered=True)
    assert np.allclose(centered_cov, expected_centered_cov)
    assert np.allclose(centered_shrinkage, expected_centered_shrinkage)
