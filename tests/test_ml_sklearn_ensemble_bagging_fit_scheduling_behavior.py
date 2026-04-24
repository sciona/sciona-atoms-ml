from __future__ import annotations

import numpy as np
from sklearn.utils import check_random_state

from sciona.atoms.ml.sklearn.ensemble.bagging_fit_scheduling import (
    bagging_fit_seeds,
    bagging_partition_estimators,
)


def test_partition_estimators_balances_remainder_to_earliest_jobs() -> None:
    n_jobs, counts, starts = bagging_partition_estimators(10, 3)
    assert n_jobs == 3
    assert np.array_equal(counts, np.array([4, 3, 3], dtype=np.int64))
    assert np.array_equal(starts, np.array([0, 4, 7, 10], dtype=np.int64))


def test_partition_estimators_caps_jobs_at_estimator_count() -> None:
    n_jobs, counts, starts = bagging_partition_estimators(2, 8)
    assert n_jobs == 2
    assert np.array_equal(counts, np.array([1, 1], dtype=np.int64))
    assert np.array_equal(starts, np.array([0, 1, 2], dtype=np.int64))


def test_fit_seeds_matches_manual_rng_advancement() -> None:
    expected_rng = check_random_state(7)
    expected_rng.randint(np.iinfo(np.int32).max, size=3)
    expected = np.asarray(
        expected_rng.randint(np.iinfo(np.int32).max, size=5),
        dtype=np.int64,
    )

    observed = bagging_fit_seeds(5, 3, random_state=7)

    assert np.array_equal(observed, expected)


def test_fit_seeds_without_previous_estimators_starts_immediately() -> None:
    expected = np.asarray(
        check_random_state(11).randint(np.iinfo(np.int32).max, size=4),
        dtype=np.int64,
    )

    observed = bagging_fit_seeds(4, 0, random_state=11)

    assert np.array_equal(observed, expected)
