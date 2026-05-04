from __future__ import annotations

import numpy as np
from sklearn.covariance import EmpiricalCovariance
from sklearn.metrics import pairwise_distances

from sciona.atoms.ml.sklearn.covariance import empirical_covariance, empirical_covariance_fit
from sciona.atoms.ml.sklearn.covariance.covariance_postfit_api_shell import (
    covariance_mahalanobis_location_row,
    covariance_mahalanobis_result,
    covariance_precision_matrix,
    covariance_score_test_covariance,
)
from sciona.atoms.ml.sklearn.covariance.state_models import CovarianceState


def test_covariance_postfit_api_shell_atoms_import() -> None:
    assert callable(covariance_precision_matrix)
    assert callable(covariance_score_test_covariance)
    assert callable(covariance_mahalanobis_location_row)
    assert callable(covariance_mahalanobis_result)


def test_covariance_precision_matrix_matches_empirical_covariance_storage_modes() -> None:
    X = np.array([[1.0, 2.0], [2.0, 3.5], [4.0, 6.0], [3.5, 5.0]], dtype=np.float64)
    stored_state = empirical_covariance_fit(X, store_precision=True, assume_centered=False)
    derived_state = empirical_covariance_fit(X, store_precision=False, assume_centered=False)

    stored_model = EmpiricalCovariance(store_precision=True).fit(X)
    derived_model = EmpiricalCovariance(store_precision=False).fit(X)

    assert np.allclose(covariance_precision_matrix(stored_state), stored_model.get_precision())
    assert np.allclose(covariance_precision_matrix(derived_state), derived_model.get_precision())


def test_covariance_score_test_covariance_matches_empirical_covariance_score_prelude() -> None:
    X_train = np.array([[1.0, 2.0], [2.0, 3.5], [4.0, 6.0], [3.5, 5.0]], dtype=np.float64)
    X_test = np.array([[2.5, 4.0], [1.5, 2.5], [3.0, 4.5]], dtype=np.float64)
    model = EmpiricalCovariance().fit(X_train)

    got = covariance_score_test_covariance(X_test, model.location_)
    expected = empirical_covariance(X_test - model.location_, assume_centered=True)

    assert np.allclose(got, expected)


def test_covariance_mahalanobis_helpers_match_empirical_covariance() -> None:
    X_train = np.array([[1.0, 2.0], [2.0, 3.5], [4.0, 6.0], [3.5, 5.0]], dtype=np.float64)
    X_test = np.array([[2.5, 4.0], [1.5, 2.5], [3.0, 4.5]], dtype=np.float64)
    model = EmpiricalCovariance().fit(X_train)

    location_row = covariance_mahalanobis_location_row(model.location_)
    pairwise = pairwise_distances(X_test, location_row, metric="mahalanobis", VI=model.get_precision())
    got = covariance_mahalanobis_result(pairwise)

    assert np.allclose(location_row, model.location_[np.newaxis, :])
    assert np.allclose(got, model.mahalanobis(X_test))


def test_covariance_postfit_api_shell_contracts() -> None:
    state = CovarianceState(
        covariance=np.eye(2, dtype=np.float64),
        location=np.zeros(2, dtype=np.float64),
        precision=None,
        store_precision=True,
        assume_centered=False,
        estimator="empirical_covariance",
        shrinkage=None,
        n_features_in=2,
    )

    try:
        covariance_precision_matrix(state)
    except Exception:
        pass
    else:
        raise AssertionError("expected invalid stored-precision state to fail")

    try:
        covariance_score_test_covariance(np.ones((2, 3)), np.ones(2))
    except Exception:
        pass
    else:
        raise AssertionError("expected mismatched location width to fail")

    try:
        covariance_mahalanobis_result(np.ones((3, 2)))
    except Exception:
        pass
    else:
        raise AssertionError("expected multi-column distances to fail")
