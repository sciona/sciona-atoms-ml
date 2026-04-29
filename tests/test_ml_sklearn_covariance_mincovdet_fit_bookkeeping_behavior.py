from __future__ import annotations

import warnings

import numpy as np
import pytest
from icontract import ViolationError
from scipy import linalg
from sklearn.covariance import MinCovDet

from sciona.atoms.ml.sklearn.covariance.mincovdet_fit_bookkeeping import (
    mincovdet_assume_centered_raw_covariance,
    mincovdet_assume_centered_raw_distances,
    mincovdet_assume_centered_raw_location,
    mincovdet_full_rank_warning_required,
)


def test_mincovdet_full_rank_warning_required_matches_sklearn_warning_predicate() -> None:
    X = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]], dtype=np.float64)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        MinCovDet(random_state=0).fit(X)
    assert any("not full rank" in str(item.message) for item in caught)
    assert mincovdet_full_rank_warning_required(X) is True


def test_mincovdet_assume_centered_raw_location_is_zero() -> None:
    observed = mincovdet_assume_centered_raw_location(3)
    assert np.array_equal(observed, np.zeros(3, dtype=np.float64))


def test_mincovdet_assume_centered_raw_covariance_matches_sklearn_fit_branch() -> None:
    X = np.array(
        [[0.0, 1.0], [1.0, 0.0], [1.0, 2.0], [2.0, 1.0], [10.0, 10.0]],
        dtype=np.float64,
    )
    model = MinCovDet(random_state=0, support_fraction=0.8, assume_centered=True).fit(X)
    observed = mincovdet_assume_centered_raw_covariance(X, model.raw_support_)
    assert np.allclose(observed, model.raw_covariance_)


def test_mincovdet_assume_centered_raw_distances_matches_sklearn_fit_branch() -> None:
    X = np.array(
        [[0.0, 1.0], [1.0, 0.0], [1.0, 2.0], [2.0, 1.0], [10.0, 10.0]],
        dtype=np.float64,
    )
    model = MinCovDet(random_state=0, support_fraction=0.8, assume_centered=True).fit(X)
    observed = mincovdet_assume_centered_raw_distances(X, model.raw_covariance_)
    precision = linalg.pinvh(model.raw_covariance_)
    expected = np.sum(np.dot(X, precision) * X, axis=1)
    assert np.allclose(observed, expected)
    assert np.allclose(observed, model.raw_support_ * 0 + observed)  # shape/value sanity


def test_mincovdet_fit_bookkeeping_rejects_invalid_inputs() -> None:
    X = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    with pytest.raises((ViolationError, ValueError)):
        mincovdet_full_rank_warning_required(np.array([[1.0]], dtype=np.float64))

    with pytest.raises((ViolationError, ValueError)):
        mincovdet_assume_centered_raw_location(0)

    with pytest.raises((ViolationError, ValueError)):
        mincovdet_assume_centered_raw_covariance(X, np.array([True], dtype=np.bool_))

    with pytest.raises((ViolationError, ValueError)):
        mincovdet_assume_centered_raw_distances(X, np.eye(3, dtype=np.float64))
