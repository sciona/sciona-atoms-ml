from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError
from sklearn.covariance._robust_covariance import select_candidates


def test_fastmcd_candidates_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.covariance.robust_fastmcd_candidates import (
        fast_mcd_candidate_pool_from_estimates,
        fast_mcd_candidate_pool_from_random_starts,
    )

    assert callable(fast_mcd_candidate_pool_from_random_starts)
    assert callable(fast_mcd_candidate_pool_from_estimates)


def test_fast_mcd_candidate_pool_from_random_starts_matches_sklearn_select_candidates_trials() -> None:
    from sciona.atoms.ml.sklearn.covariance.robust_fastmcd_candidates import fast_mcd_candidate_pool_from_random_starts

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
    result = fast_mcd_candidate_pool_from_random_starts(X, 4, 3, n_iter=5, random_state=7)
    expected = select_candidates(X, 4, 3, select=3, n_iter=5, random_state=7)

    assert np.allclose(result[0], expected[0])
    assert np.allclose(result[1], expected[1])
    assert np.array_equal(result[2], np.linalg.slogdet(expected[1])[1])
    assert np.array_equal(result[3], expected[2])
    assert np.allclose(result[4], expected[3])


def test_fast_mcd_candidate_pool_from_estimates_matches_sklearn_select_candidates_trials() -> None:
    from sciona.atoms.ml.sklearn.covariance.robust_fastmcd_candidates import fast_mcd_candidate_pool_from_estimates

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
    initial_locations = np.array(
        [
            X[:4].mean(axis=0),
            X[1:5].mean(axis=0),
        ],
        dtype=np.float64,
    )
    initial_covariances = np.array(
        [
            np.cov(X[:4].T, bias=1),
            np.cov(X[1:5].T, bias=1),
        ],
        dtype=np.float64,
    )
    result = fast_mcd_candidate_pool_from_estimates(
        X,
        initial_locations,
        initial_covariances,
        4,
        n_iter=5,
        random_state=5,
    )
    expected = select_candidates(
        X,
        4,
        (initial_locations, initial_covariances),
        select=2,
        n_iter=5,
        random_state=5,
    )

    assert np.allclose(result[0], expected[0])
    assert np.allclose(result[1], expected[1])
    assert np.array_equal(result[2], np.linalg.slogdet(expected[1])[1])
    assert np.array_equal(result[3], expected[2])
    assert np.allclose(result[4], expected[3])


def test_contracts_reject_invalid_fastmcd_candidates_inputs() -> None:
    from sciona.atoms.ml.sklearn.covariance.robust_fastmcd_candidates import (
        fast_mcd_candidate_pool_from_estimates,
        fast_mcd_candidate_pool_from_random_starts,
    )

    X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)

    with pytest.raises(ViolationError):
        fast_mcd_candidate_pool_from_random_starts(X, 3, 1)

    with pytest.raises(ViolationError):
        fast_mcd_candidate_pool_from_estimates(
            X,
            np.ones((1, 1), dtype=np.float64),
            np.ones((1, 2, 2), dtype=np.float64),
            1,
        )
