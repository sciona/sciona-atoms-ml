from __future__ import annotations

import numpy as np
from scipy import linalg
from sklearn.covariance import EmpiricalCovariance, LedoitWolf, OAS, ShrunkCovariance


def test_covariance_fit_bookkeeping_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.covariance.covariance_fit_bookkeeping_shell import (
        covariance_fit_location,
        covariance_set_covariance_matrix,
        covariance_set_precision_matrix,
        covariance_set_precision_required,
    )

    assert callable(covariance_fit_location)
    assert callable(covariance_set_covariance_matrix)
    assert callable(covariance_set_precision_required)
    assert callable(covariance_set_precision_matrix)


def test_covariance_fit_location_matches_sklearn_estimators() -> None:
    from sciona.atoms.ml.sklearn.covariance.covariance_fit_bookkeeping_shell import covariance_fit_location

    X = np.array([[1.0, 2.0], [2.0, 3.5], [4.0, 6.0], [3.5, 5.0]], dtype=np.float64)

    assert np.allclose(covariance_fit_location(X, assume_centered=False), EmpiricalCovariance().fit(X).location_)
    assert np.allclose(
        covariance_fit_location(X, assume_centered=True),
        EmpiricalCovariance(assume_centered=True).fit(X).location_,
    )
    assert np.allclose(covariance_fit_location(X, assume_centered=False), ShrunkCovariance().fit(X).location_)
    assert np.allclose(covariance_fit_location(X, assume_centered=False), LedoitWolf().fit(X).location_)
    assert np.allclose(covariance_fit_location(X, assume_centered=False), OAS().fit(X).location_)


def test_covariance_set_covariance_and_precision_match_sklearn_set_covariance() -> None:
    from sciona.atoms.ml.sklearn.covariance.covariance_fit_bookkeeping_shell import (
        covariance_set_covariance_matrix,
        covariance_set_precision_matrix,
        covariance_set_precision_required,
    )

    covariance = np.array([[2.0, 0.5], [0.5, 1.25]], dtype=np.float64)
    estimator = EmpiricalCovariance(store_precision=True)
    estimator._set_covariance(covariance)

    assert np.array_equal(covariance_set_covariance_matrix(covariance), estimator.covariance_)
    assert covariance_set_precision_required(True) is True
    assert covariance_set_precision_required(False) is False
    assert np.allclose(covariance_set_precision_matrix(covariance), estimator.precision_)
    assert np.allclose(covariance_set_precision_matrix(covariance), linalg.pinvh(covariance, check_finite=False))


def test_covariance_fit_bookkeeping_shell_contracts() -> None:
    from sciona.atoms.ml.sklearn.covariance.covariance_fit_bookkeeping_shell import (
        covariance_fit_location,
        covariance_set_covariance_matrix,
        covariance_set_precision_required,
    )

    try:
        covariance_fit_location(np.array([1.0, 2.0]), assume_centered=False)
    except Exception:
        pass
    else:
        raise AssertionError("expected 1D input to fail")

    try:
        covariance_set_covariance_matrix(np.ones((2, 3)))
    except Exception:
        pass
    else:
        raise AssertionError("expected nonsquare covariance to fail")

    try:
        covariance_set_precision_required(1)
    except Exception:
        pass
    else:
        raise AssertionError("expected nonboolean store_precision to fail")
