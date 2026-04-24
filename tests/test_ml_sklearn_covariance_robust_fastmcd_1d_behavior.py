from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError
from sklearn.covariance._robust_covariance import fast_mcd


def _one_dimensional_data() -> np.ndarray:
    return np.array([[-3.0], [-1.0], [0.0], [0.5], [1.0], [10.0]], dtype=np.float64)


def _tiny_one_dimensional_data() -> np.ndarray:
    return np.array([[-1.0], [0.0], [2.0]], dtype=np.float64)


def test_robust_fastmcd_1d_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.covariance.robust_fastmcd_1d import (
        fast_mcd_1d_covariance,
        fast_mcd_1d_location,
        fast_mcd_1d_squared_distances,
        fast_mcd_1d_support_mask,
        fast_mcd_support_count,
    )

    assert callable(fast_mcd_support_count)
    assert callable(fast_mcd_1d_location)
    assert callable(fast_mcd_1d_support_mask)
    assert callable(fast_mcd_1d_covariance)
    assert callable(fast_mcd_1d_squared_distances)


def test_fastmcd_support_count_matches_sklearn_resolution() -> None:
    from sciona.atoms.ml.sklearn.covariance.robust_fastmcd_1d import fast_mcd_support_count

    X = _one_dimensional_data()
    assert fast_mcd_support_count(X.shape[0], X.shape[1]) == 4
    assert fast_mcd_support_count(X.shape[0], X.shape[1], support_fraction=0.75) == 4


def test_fastmcd_1d_helpers_match_sklearn_private_function_shortest_half_branch() -> None:
    from sciona.atoms.ml.sklearn.covariance.robust_fastmcd_1d import (
        fast_mcd_1d_covariance,
        fast_mcd_1d_location,
        fast_mcd_1d_squared_distances,
        fast_mcd_1d_support_mask,
        fast_mcd_support_count,
    )

    X = _one_dimensional_data()
    n_support = fast_mcd_support_count(X.shape[0], X.shape[1])
    expected_location, expected_covariance, expected_support, expected_dist = fast_mcd(X)

    actual_location = fast_mcd_1d_location(X, n_support)
    actual_support = fast_mcd_1d_support_mask(X, actual_location, n_support)
    actual_covariance = fast_mcd_1d_covariance(X, actual_support)
    actual_dist = fast_mcd_1d_squared_distances(X, actual_location, actual_covariance)

    assert np.allclose(actual_location, expected_location)
    assert np.array_equal(actual_support, expected_support)
    assert np.allclose(actual_covariance, expected_covariance)
    assert np.allclose(actual_dist, expected_dist)


def test_fastmcd_1d_helpers_match_sklearn_private_function_full_support_branch() -> None:
    from sciona.atoms.ml.sklearn.covariance.robust_fastmcd_1d import (
        fast_mcd_1d_covariance,
        fast_mcd_1d_location,
        fast_mcd_1d_squared_distances,
        fast_mcd_1d_support_mask,
        fast_mcd_support_count,
    )

    X = _tiny_one_dimensional_data()
    n_support = fast_mcd_support_count(X.shape[0], X.shape[1])
    expected_location, expected_covariance, expected_support, expected_dist = fast_mcd(X)

    actual_location = fast_mcd_1d_location(X, n_support)
    actual_support = fast_mcd_1d_support_mask(X, actual_location, n_support)
    actual_covariance = fast_mcd_1d_covariance(X, actual_support)
    actual_dist = fast_mcd_1d_squared_distances(X, actual_location, actual_covariance)

    assert n_support == X.shape[0]
    assert np.allclose(actual_location, expected_location)
    assert np.array_equal(actual_support, expected_support)
    assert np.allclose(actual_covariance, expected_covariance)
    assert np.allclose(actual_dist, expected_dist)


def test_contracts_reject_invalid_fastmcd_1d_inputs() -> None:
    from sciona.atoms.ml.sklearn.covariance.robust_fastmcd_1d import (
        fast_mcd_1d_covariance,
        fast_mcd_1d_location,
        fast_mcd_1d_squared_distances,
        fast_mcd_1d_support_mask,
        fast_mcd_support_count,
    )

    X = _one_dimensional_data()

    with pytest.raises(ViolationError):
        fast_mcd_support_count(0, 1)

    with pytest.raises(ViolationError):
        fast_mcd_1d_location(np.array([[1.0]], dtype=np.float64), 1)

    with pytest.raises(ViolationError):
        fast_mcd_1d_support_mask(X, np.array([0.0], dtype=np.float64), 7)

    with pytest.raises(ViolationError):
        fast_mcd_1d_covariance(X, np.zeros(X.shape[0], dtype=np.bool_))

    with pytest.raises(ViolationError):
        fast_mcd_1d_squared_distances(X, np.array([0.0], dtype=np.float64), np.array([[np.nan]], dtype=np.float64))
