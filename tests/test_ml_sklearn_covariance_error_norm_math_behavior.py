from __future__ import annotations

import numpy as np
import pytest
from sklearn.covariance import EmpiricalCovariance


def test_covariance_error_norm_math_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.covariance.error_norm_math import (
        covariance_error_matrix,
        covariance_error_result,
        covariance_error_scaled_squared_norm,
        covariance_error_squared_norm,
    )

    assert callable(covariance_error_matrix)
    assert callable(covariance_error_squared_norm)
    assert callable(covariance_error_scaled_squared_norm)
    assert callable(covariance_error_result)


def test_covariance_error_norm_math_matches_source_logic() -> None:
    from sciona.atoms.ml.sklearn.covariance.error_norm_math import (
        covariance_error_matrix,
        covariance_error_result,
        covariance_error_scaled_squared_norm,
        covariance_error_squared_norm,
    )

    comp_cov = np.array([[2.0, 0.5], [0.5, 1.5]], dtype=np.float64)
    covariance = np.array([[1.0, 0.25], [0.25, 1.0]], dtype=np.float64)
    error = covariance_error_matrix(comp_cov, covariance)
    assert np.array_equal(error, comp_cov - covariance)

    fro_sq = covariance_error_squared_norm(error, "frobenius")
    assert fro_sq == pytest.approx(np.sum(error**2))

    spectral_sq = covariance_error_squared_norm(error, "spectral")
    assert spectral_sq == pytest.approx(np.amax(np.linalg.svd(error.T @ error, compute_uv=False)))

    scaled = covariance_error_scaled_squared_norm(fro_sq, scaling=True, n_features=2)
    assert scaled == pytest.approx(fro_sq / 2.0)
    assert covariance_error_result(scaled, squared=True) == pytest.approx(scaled)
    assert covariance_error_result(scaled, squared=False) == pytest.approx(np.sqrt(scaled))


def test_covariance_error_norm_math_matches_empirical_covariance_error_norm() -> None:
    from sciona.atoms.ml.sklearn.covariance.error_norm_math import (
        covariance_error_matrix,
        covariance_error_result,
        covariance_error_scaled_squared_norm,
        covariance_error_squared_norm,
    )

    X = np.array([[1.0, 2.0], [2.0, 3.5], [4.0, 6.0], [3.5, 5.0]], dtype=np.float64)
    model = EmpiricalCovariance().fit(X)
    comp_cov = np.array([[1.5, 0.2], [0.2, 2.0]], dtype=np.float64)

    error = covariance_error_matrix(comp_cov, model.covariance_)
    got_default = covariance_error_result(
        covariance_error_scaled_squared_norm(
            covariance_error_squared_norm(error, "frobenius"),
            scaling=True,
            n_features=error.shape[0],
        ),
        squared=True,
    )
    assert got_default == pytest.approx(model.error_norm(comp_cov))

    got_spectral = covariance_error_result(
        covariance_error_scaled_squared_norm(
            covariance_error_squared_norm(error, "spectral"),
            scaling=False,
            n_features=error.shape[0],
        ),
        squared=False,
    )
    assert got_spectral == pytest.approx(
        model.error_norm(comp_cov, norm="spectral", scaling=False, squared=False)
    )


def test_covariance_error_norm_math_invalid_norm_and_contracts() -> None:
    from sciona.atoms.ml.sklearn.covariance.error_norm_math import (
        covariance_error_matrix,
        covariance_error_result,
        covariance_error_scaled_squared_norm,
        covariance_error_squared_norm,
    )

    error = np.eye(2, dtype=np.float64)

    with pytest.raises(NotImplementedError):
        covariance_error_squared_norm(error, "bad")

    with pytest.raises(Exception):
        covariance_error_matrix(np.eye(2), np.ones((3, 3)))

    with pytest.raises(Exception):
        covariance_error_scaled_squared_norm(-1.0, True, 2)

    with pytest.raises(Exception):
        covariance_error_result(-1.0, False)
