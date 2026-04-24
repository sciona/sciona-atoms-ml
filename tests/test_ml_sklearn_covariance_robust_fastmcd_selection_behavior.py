from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError


def test_robust_fastmcd_selection_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.covariance.robust_fastmcd_selection import (
        fast_mcd_best_candidate_indices,
        fast_mcd_gather_best_candidates,
        fast_mcd_large_sample_schedule,
        fast_mcd_place_merged_results,
        fast_mcd_trial_plan,
    )

    assert callable(fast_mcd_trial_plan)
    assert callable(fast_mcd_best_candidate_indices)
    assert callable(fast_mcd_gather_best_candidates)
    assert callable(fast_mcd_large_sample_schedule)
    assert callable(fast_mcd_place_merged_results)


def test_fastmcd_trial_plan_matches_integer_and_estimate_tuple_modes() -> None:
    from sciona.atoms.ml.sklearn.covariance.robust_fastmcd_selection import fast_mcd_trial_plan

    locations = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float64)
    covariances = np.array(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[2.0, 0.5], [0.5, 3.0]],
        ],
        dtype=np.float64,
    )

    assert fast_mcd_trial_plan(7) == (False, 7)
    assert fast_mcd_trial_plan((locations, covariances)) == (True, 2)


def test_fastmcd_trial_plan_reuses_sklearn_type_error_message_for_invalid_types() -> None:
    from sciona.atoms.ml.sklearn.covariance.robust_fastmcd_selection import fast_mcd_trial_plan

    with pytest.raises(TypeError, match=r"Invalid 'n_trials' parameter, expected tuple or  integer"):
        fast_mcd_trial_plan("bad-input")


def test_fastmcd_best_candidate_indices_match_numpy_argsort_prefix() -> None:
    from sciona.atoms.ml.sklearn.covariance.robust_fastmcd_selection import fast_mcd_best_candidate_indices

    determinants = np.array([5.0, 1.5, 3.0, 0.75], dtype=np.float64)
    actual = fast_mcd_best_candidate_indices(determinants, select=2)

    assert np.array_equal(actual, np.array([3, 1], dtype=np.int64))


def test_fastmcd_gather_best_candidates_selects_ranked_rows() -> None:
    from sciona.atoms.ml.sklearn.covariance.robust_fastmcd_selection import (
        fast_mcd_best_candidate_indices,
        fast_mcd_gather_best_candidates,
    )

    determinants = np.array([4.0, 0.5, 2.0], dtype=np.float64)
    locations = np.array([[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]], dtype=np.float64)
    covariances = np.array(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[2.0, 0.1], [0.1, 2.0]],
            [[3.0, 0.2], [0.2, 3.0]],
        ],
        dtype=np.float64,
    )
    supports = np.array(
        [
            [True, False, True],
            [False, True, True],
            [True, True, False],
        ],
        dtype=np.bool_,
    )
    distances = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ],
        dtype=np.float64,
    )

    indices = fast_mcd_best_candidate_indices(determinants, select=2)
    best_locations, best_covariances, best_supports, best_distances = fast_mcd_gather_best_candidates(
        locations,
        covariances,
        supports,
        distances,
        indices,
    )

    assert np.array_equal(indices, np.array([1, 2], dtype=np.int64))
    assert np.array_equal(best_locations, locations[[1, 2]])
    assert np.array_equal(best_covariances, covariances[[1, 2]])
    assert np.array_equal(best_supports, supports[[1, 2]])
    assert np.array_equal(best_distances, distances[[1, 2]])


def test_fastmcd_large_sample_schedule_matches_sklearn_formulae() -> None:
    from sciona.atoms.ml.sklearn.covariance.robust_fastmcd_selection import fast_mcd_large_sample_schedule

    actual = fast_mcd_large_sample_schedule(n_samples=1200, n_features=4, n_support=650)

    assert actual == (4, 300, 163, 500, 10, 125, 40, 1200, 650, 1)


def test_fastmcd_large_sample_schedule_uses_large_dataset_merged_branch_when_needed() -> None:
    from sciona.atoms.ml.sklearn.covariance.robust_fastmcd_selection import fast_mcd_large_sample_schedule

    actual = fast_mcd_large_sample_schedule(n_samples=2400, n_features=3, n_support=1201)

    assert actual == (8, 300, 151, 500, 10, 62, 80, 1500, 751, 10)


def test_fastmcd_place_merged_results_matches_full_length_scatter() -> None:
    from sciona.atoms.ml.sklearn.covariance.robust_fastmcd_selection import fast_mcd_place_merged_results

    selection = np.array([4, 1, 3], dtype=np.int64)
    merged_support = np.array([True, False, True], dtype=np.bool_)
    merged_distances = np.array([0.1, 0.2, 0.3], dtype=np.float64)

    support, distances = fast_mcd_place_merged_results(
        6,
        selection,
        merged_support,
        merged_distances,
    )

    expected_support = np.array([False, False, False, True, True, False], dtype=np.bool_)
    expected_distances = np.array([0.0, 0.2, 0.0, 0.3, 0.1, 0.0], dtype=np.float64)
    assert np.array_equal(support, expected_support)
    assert np.allclose(distances, expected_distances)


def test_contracts_reject_invalid_fastmcd_selection_inputs() -> None:
    from sciona.atoms.ml.sklearn.covariance.robust_fastmcd_selection import (
        fast_mcd_best_candidate_indices,
        fast_mcd_gather_best_candidates,
        fast_mcd_large_sample_schedule,
        fast_mcd_place_merged_results,
        fast_mcd_trial_plan,
    )

    with pytest.raises(ValueError):
        fast_mcd_trial_plan(0)

    with pytest.raises(ViolationError):
        fast_mcd_best_candidate_indices(np.array([1.0, 2.0], dtype=np.float64), select=3)

    with pytest.raises(ViolationError):
        fast_mcd_gather_best_candidates(
            np.ones((2, 2), dtype=np.float64),
            np.ones((3, 2, 2), dtype=np.float64),
            np.ones((2, 4), dtype=np.bool_),
            np.ones((2, 4), dtype=np.float64),
            np.array([0], dtype=np.int64),
        )

    with pytest.raises(ViolationError):
        fast_mcd_large_sample_schedule(500, 2, 200)

    with pytest.raises(ViolationError):
        fast_mcd_place_merged_results(
            4,
            np.array([0, 4], dtype=np.int64),
            np.array([True, False], dtype=np.bool_),
            np.array([1.0, 2.0], dtype=np.float64),
        )
