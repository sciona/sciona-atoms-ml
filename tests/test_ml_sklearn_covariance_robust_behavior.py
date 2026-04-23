from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError
from scipy.stats import chi2
from sklearn.covariance import MinCovDet
from sklearn.covariance._robust_covariance import _consistency_factor


def _data() -> np.ndarray:
    return np.array(
        [
            [-1.2, 0.1],
            [-0.8, -0.2],
            [-0.3, 0.4],
            [0.2, -0.1],
            [0.7, 0.5],
            [1.1, 0.2],
            [4.0, 4.2],
            [-3.5, 3.8],
        ],
        dtype=np.float64,
    )


def test_robust_covariance_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.covariance.robust import (
        mcd_consistency_factor,
        mcd_correct_covariance,
        mcd_reweight_support_mask,
        mcd_reweighted_location_covariance,
        mcd_squared_mahalanobis,
    )

    assert callable(mcd_consistency_factor)
    assert callable(mcd_correct_covariance)
    assert callable(mcd_reweight_support_mask)
    assert callable(mcd_reweighted_location_covariance)
    assert callable(mcd_squared_mahalanobis)


def test_mcd_consistency_factor_matches_sklearn_private_helper() -> None:
    from sciona.atoms.ml.sklearn.covariance.robust import mcd_consistency_factor

    for n_features, alpha in [(2, 0.5), (3, 0.75), (5, 0.975)]:
        assert np.isclose(mcd_consistency_factor(n_features, alpha), _consistency_factor(n_features, alpha))


def test_mcd_correct_covariance_matches_sklearn_min_cov_det_method() -> None:
    from sciona.atoms.ml.sklearn.covariance.robust import mcd_correct_covariance

    X = _data()
    model = MinCovDet(random_state=0, support_fraction=0.75).fit(X)
    raw_covariance = model.raw_covariance_.copy()
    raw_dist = model.dist_.copy()
    n_support = int(np.sum(model.support_))

    expected_covariance = model.correct_covariance(X)
    actual_covariance, actual_dist = mcd_correct_covariance(raw_covariance, raw_dist, n_support)

    assert np.allclose(actual_covariance, expected_covariance)
    assert np.allclose(actual_dist, model.dist_)


def test_mcd_reweighting_matches_sklearn_min_cov_det_method() -> None:
    from sciona.atoms.ml.sklearn.covariance.robust import (
        mcd_reweight_support_mask,
        mcd_reweighted_location_covariance,
    )

    X = _data()
    model = MinCovDet(random_state=0, support_fraction=0.75).fit(X)
    dist = model.dist_.copy()
    expected_location, expected_covariance, expected_support = model.reweight_covariance(X)

    mask = mcd_reweight_support_mask(dist, X.shape[1])
    actual_location, actual_covariance, actual_support = mcd_reweighted_location_covariance(X, mask)

    assert np.array_equal(mask, dist < chi2(X.shape[1]).isf(0.025))
    assert np.array_equal(actual_support, expected_support)
    assert np.allclose(actual_location, expected_location)
    assert np.allclose(actual_covariance, expected_covariance)


def test_mcd_reweighting_assume_centered_matches_sklearn() -> None:
    from sciona.atoms.ml.sklearn.covariance.robust import (
        mcd_reweight_support_mask,
        mcd_reweighted_location_covariance,
    )

    X = _data()
    model = MinCovDet(random_state=0, support_fraction=0.75, assume_centered=True).fit(X)
    dist = model.dist_.copy()
    expected_location, expected_covariance, expected_support = model.reweight_covariance(X)

    mask = mcd_reweight_support_mask(dist, X.shape[1])
    actual_location, actual_covariance, actual_support = mcd_reweighted_location_covariance(X, mask, assume_centered=True)

    assert np.array_equal(actual_support, expected_support)
    assert np.allclose(actual_location, expected_location)
    assert np.allclose(actual_covariance, expected_covariance)


def test_mcd_squared_mahalanobis_matches_sklearn_method() -> None:
    from sciona.atoms.ml.sklearn.covariance.robust import mcd_squared_mahalanobis

    X = _data()
    model = MinCovDet(random_state=0, support_fraction=0.75).fit(X)
    actual = mcd_squared_mahalanobis(X, model.location_, model.get_precision())

    assert np.allclose(actual, model.mahalanobis(X))


def test_contracts_reject_invalid_robust_covariance_inputs() -> None:
    from sciona.atoms.ml.sklearn.covariance.robust import (
        mcd_consistency_factor,
        mcd_correct_covariance,
        mcd_reweighted_location_covariance,
        mcd_squared_mahalanobis,
    )

    X = _data()

    with pytest.raises(ViolationError):
        mcd_consistency_factor(0, 0.5)

    with pytest.raises(ViolationError):
        mcd_correct_covariance(np.ones((2, 3), dtype=np.float64), np.ones(4, dtype=np.float64), 2)

    with pytest.raises(ViolationError):
        mcd_correct_covariance(np.eye(2), np.ones(4, dtype=np.float64), 5)

    with pytest.raises(ViolationError):
        mcd_reweighted_location_covariance(X, np.zeros(X.shape[0], dtype=np.bool_))

    with pytest.raises(ViolationError):
        mcd_squared_mahalanobis(X, np.zeros(3, dtype=np.float64), np.eye(2))
