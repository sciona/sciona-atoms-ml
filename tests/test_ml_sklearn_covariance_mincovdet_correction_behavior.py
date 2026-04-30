from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError
from sklearn.covariance import MinCovDet
from sklearn.covariance._robust_covariance import fast_mcd


def test_mincovdet_correction_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.covariance.mincovdet_correction import (
        mincovdet_correct_covariance_guard,
        mincovdet_corrected_covariance,
        mincovdet_corrected_distances,
        mincovdet_empirical_correction_factor,
    )

    assert callable(mincovdet_correct_covariance_guard)
    assert callable(mincovdet_empirical_correction_factor)
    assert callable(mincovdet_corrected_covariance)
    assert callable(mincovdet_corrected_distances)


def test_mincovdet_correction_matches_sklearn_correct_covariance_method() -> None:
    from sciona.atoms.ml.sklearn.covariance.mincovdet_correction import (
        mincovdet_correct_covariance_guard,
        mincovdet_corrected_covariance,
        mincovdet_corrected_distances,
        mincovdet_empirical_correction_factor,
    )

    X = np.array(
        [[0.0, 1.0], [1.0, 0.0], [1.0, 2.0], [2.0, 1.0], [10.0, 10.0]],
        dtype=np.float64,
    )
    raw_location, raw_covariance, raw_support, raw_distances = fast_mcd(
        X,
        support_fraction=0.8,
        cov_computation_method=lambda values, assume_centered=False: np.cov(
            values.T, bias=True
        )
        if not assume_centered
        else np.dot(values.T, values) / values.shape[0],
        random_state=np.random.RandomState(0),
    )
    del raw_location

    model = MinCovDet(random_state=0, support_fraction=0.8)
    model.raw_covariance_ = np.asarray(raw_covariance, dtype=np.float64)
    model.support_ = np.asarray(raw_support, dtype=np.bool_)
    model.dist_ = np.asarray(raw_distances, dtype=np.float64).copy()
    expected_covariance = model.correct_covariance(X)
    expected_distances = np.asarray(model.dist_, dtype=np.float64)

    assert mincovdet_correct_covariance_guard(raw_covariance, raw_support, raw_distances) is True
    factor = mincovdet_empirical_correction_factor(raw_distances, X.shape[1])
    observed_covariance = mincovdet_corrected_covariance(raw_covariance, factor)
    observed_distances = mincovdet_corrected_distances(raw_distances, factor)

    assert np.allclose(observed_covariance, expected_covariance)
    assert np.allclose(observed_distances, expected_distances)


def test_mincovdet_correct_covariance_guard_matches_sklearn_zero_covariance_error() -> None:
    from sciona.atoms.ml.sklearn.covariance.mincovdet_correction import (
        mincovdet_correct_covariance_guard,
    )

    raw_covariance = np.zeros((2, 2), dtype=np.float64)
    raw_support = np.array([True, False, True], dtype=np.bool_)
    raw_distances = np.array([0.0, 0.0, 0.0], dtype=np.float64)

    with pytest.raises(ValueError, match="support data is equal to 0"):
        mincovdet_correct_covariance_guard(raw_covariance, raw_support, raw_distances)


def test_mincovdet_correction_contracts_reject_invalid_inputs() -> None:
    from sciona.atoms.ml.sklearn.covariance.mincovdet_correction import (
        mincovdet_correct_covariance_guard,
        mincovdet_corrected_covariance,
        mincovdet_corrected_distances,
        mincovdet_empirical_correction_factor,
    )

    with pytest.raises(ViolationError):
        mincovdet_correct_covariance_guard(
            np.eye(2, dtype=np.float64),
            np.array([True], dtype=np.bool_),
            np.array([1.0, 2.0], dtype=np.float64),
        )

    with pytest.raises(ViolationError):
        mincovdet_empirical_correction_factor(np.array([-1.0], dtype=np.float64), 2)

    with pytest.raises(ViolationError):
        mincovdet_corrected_covariance(np.eye(2, dtype=np.float64), 0.0)

    with pytest.raises(ViolationError):
        mincovdet_corrected_distances(np.array([1.0], dtype=np.float64), 0.0)
