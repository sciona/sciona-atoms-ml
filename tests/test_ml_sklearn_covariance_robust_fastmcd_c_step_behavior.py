from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError
from scipy import linalg
from sklearn.covariance import empirical_covariance as sklearn_empirical_covariance
from sklearn.covariance._robust_covariance import _c_step
from sklearn.utils import check_random_state


def test_fastmcd_c_step_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.covariance.robust_fastmcd_c_step import (
        fast_mcd_c_step,
        fast_mcd_initial_random_support_indices,
        fast_mcd_support_indices_from_estimates,
        fast_mcd_support_statistics,
    )

    assert callable(fast_mcd_initial_random_support_indices)
    assert callable(fast_mcd_support_indices_from_estimates)
    assert callable(fast_mcd_support_statistics)
    assert callable(fast_mcd_c_step)


def test_fast_mcd_initial_random_support_indices_matches_sklearn() -> None:
    from sciona.atoms.ml.sklearn.covariance.robust_fastmcd_c_step import fast_mcd_initial_random_support_indices

    expected = check_random_state(7).permutation(20)[:8]
    result = fast_mcd_initial_random_support_indices(20, 8, random_state=7)
    assert np.array_equal(result, expected)


def test_fast_mcd_support_indices_from_estimates_matches_sklearn_formula() -> None:
    from sciona.atoms.ml.sklearn.covariance.robust_fastmcd_c_step import fast_mcd_support_indices_from_estimates

    X = np.array(
        [
            [0.0, 0.2],
            [0.1, -0.1],
            [0.2, 0.1],
            [4.0, 4.1],
            [4.2, 3.9],
        ],
        dtype=np.float64,
    )
    location = np.array([0.1, 0.0], dtype=np.float64)
    covariance = sklearn_empirical_covariance(X[:3])
    precision = linalg.pinvh(covariance)
    centered = X - location
    dist = np.sum(np.dot(centered, precision) * centered, axis=1)
    expected = np.argpartition(dist, 2)[:3]

    result = fast_mcd_support_indices_from_estimates(X, location, covariance, n_support=3)
    assert np.array_equal(np.sort(result), np.sort(expected))


def test_fast_mcd_support_statistics_matches_manual_values() -> None:
    from sciona.atoms.ml.sklearn.covariance.robust_fastmcd_c_step import fast_mcd_support_statistics

    X = np.array(
        [
            [0.0, 0.2],
            [0.1, -0.1],
            [0.2, 0.1],
            [4.0, 4.1],
        ],
        dtype=np.float64,
    )
    support_indices = np.array([0, 1, 2], dtype=np.int64)
    result_location, result_covariance, result_det = fast_mcd_support_statistics(X, support_indices)
    expected_location = X[support_indices].mean(axis=0)
    expected_covariance = sklearn_empirical_covariance(X[support_indices])

    assert np.allclose(result_location, expected_location)
    assert np.allclose(result_covariance, expected_covariance)
    assert np.isclose(result_det, np.linalg.slogdet(expected_covariance)[1])


def test_fast_mcd_c_step_matches_sklearn_random_start() -> None:
    from sciona.atoms.ml.sklearn.covariance.robust_fastmcd_c_step import fast_mcd_c_step

    X = np.array(
        [
            [0.0, 0.2],
            [0.1, -0.1],
            [0.2, 0.1],
            [3.9, 4.0],
            [4.1, 3.8],
            [4.2, 4.2],
        ],
        dtype=np.float64,
    )
    expected = _c_step(X, 4, random_state=check_random_state(11))
    result = fast_mcd_c_step(X, 4, random_state=11)

    assert np.allclose(result[0], expected[0])
    assert np.allclose(result[1], expected[1])
    assert np.isclose(result[2], expected[2]) or (np.isinf(result[2]) and np.isinf(expected[2]))
    assert np.array_equal(result[3], expected[3])
    assert np.allclose(result[4], expected[4])


def test_fast_mcd_c_step_matches_sklearn_initial_estimates() -> None:
    from sciona.atoms.ml.sklearn.covariance.robust_fastmcd_c_step import fast_mcd_c_step

    X = np.array(
        [
            [0.0, 0.2],
            [0.1, -0.1],
            [0.2, 0.1],
            [0.3, -0.2],
            [4.1, 4.0],
            [4.2, 3.9],
        ],
        dtype=np.float64,
    )
    initial_location = X[:4].mean(axis=0)
    initial_covariance = sklearn_empirical_covariance(X[:4])
    expected = _c_step(
        X,
        4,
        random_state=check_random_state(3),
        initial_estimates=(initial_location, initial_covariance),
    )
    result = fast_mcd_c_step(
        X,
        4,
        random_state=3,
        initial_location=initial_location,
        initial_covariance=initial_covariance,
    )

    assert np.allclose(result[0], expected[0])
    assert np.allclose(result[1], expected[1])
    assert np.isclose(result[2], expected[2]) or (np.isinf(result[2]) and np.isinf(expected[2]))
    assert np.array_equal(result[3], expected[3])
    assert np.allclose(result[4], expected[4])


def test_contracts_reject_invalid_fastmcd_c_step_inputs() -> None:
    from sciona.atoms.ml.sklearn.covariance.robust_fastmcd_c_step import (
        fast_mcd_c_step,
        fast_mcd_initial_random_support_indices,
        fast_mcd_support_indices_from_estimates,
        fast_mcd_support_statistics,
    )

    X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
    covariance = np.eye(2, dtype=np.float64)

    with pytest.raises(ViolationError):
        fast_mcd_initial_random_support_indices(1, 1)

    with pytest.raises(ViolationError):
        fast_mcd_support_indices_from_estimates(X, np.array([0.0], dtype=np.float64), covariance, n_support=1)

    with pytest.raises(ViolationError):
        fast_mcd_support_statistics(X, np.array([0, 2], dtype=np.int64))

    with pytest.raises(ViolationError):
        fast_mcd_c_step(X, 3)
