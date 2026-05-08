"""Bagging fit scheduling helpers adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_bagging_fit_seeds,
    witness_bagging_partition_estimators,
)

MAX_INT = np.iinfo(np.int32).max
RandomStateLike = int | np.random.RandomState | None
PartitionResult = tuple[int, NDArray[np.int64], NDArray[np.int64]]

def _positive_int(value: int) -> bool:
    return bool(isinstance(value, int) and not isinstance(value, bool) and value >= 1)

def _n_jobs_valid(value: int | None) -> bool:
    return bool(value is None or (isinstance(value, int) and not isinstance(value, bool) and value != 0))

def _partition_result_valid(result: PartitionResult, n_estimators: int) -> bool:
    n_jobs, n_estimators_per_job, starts = result
    counts = np.asarray(n_estimators_per_job, dtype=np.int64)
    offsets = np.asarray(starts, dtype=np.int64)
    return bool(
        isinstance(n_jobs, int)
        and 1 <= n_jobs <= n_estimators
        and counts.shape == (n_jobs,)
        and offsets.shape == (n_jobs + 1,)
        and offsets[0] == 0
        and offsets[-1] == n_estimators
        and np.all(counts >= 1)
        and int(np.sum(counts)) == n_estimators
        and np.array_equal(offsets[1:], np.cumsum(counts))
        and int(np.max(counts) - np.min(counts)) <= 1
    )

def _seed_vector_valid(result: NDArray[np.int64], n_more_estimators: int) -> bool:
    values = np.asarray(result)
    return bool(
        values.shape == (n_more_estimators,)
        and np.issubdtype(values.dtype, np.integer)
        and np.all(values >= 0)
        and np.all(values < MAX_INT)
    )

@register_atom(witness_bagging_partition_estimators)
@icontract.require(lambda n_estimators: _positive_int(n_estimators), "n_estimators must be a positive integer")
@icontract.require(lambda n_jobs: _n_jobs_valid(n_jobs), "n_jobs must be None or a nonzero integer")
@icontract.ensure(
    lambda result, n_estimators: _partition_result_valid(result, n_estimators),
    "partition result must cover each estimator exactly once across balanced job slices",
)
def bagging_partition_estimators(
    n_estimators: int,
    n_jobs: int | None,
) -> PartitionResult:
    from joblib import effective_n_jobs
    """Partition sklearn bagging's estimator count across effective parallel jobs."""
    resolved_n_jobs = min(effective_n_jobs(n_jobs), n_estimators)
    n_estimators_per_job = np.full(resolved_n_jobs, n_estimators // resolved_n_jobs, dtype=np.int64)
    n_estimators_per_job[: n_estimators % resolved_n_jobs] += 1
    starts = np.cumsum(n_estimators_per_job, dtype=np.int64)
    return (
        int(resolved_n_jobs),
        np.asarray(n_estimators_per_job, dtype=np.int64),
        np.asarray([0, *starts.tolist()], dtype=np.int64),
    )

@register_atom(witness_bagging_fit_seeds)
@icontract.require(lambda n_more_estimators: _positive_int(n_more_estimators), "n_more_estimators must be a positive integer")
@icontract.require(
    lambda previous_estimators: isinstance(previous_estimators, int) and not isinstance(previous_estimators, bool) and previous_estimators >= 0,
    "previous_estimators must be a nonnegative integer",
)
@icontract.ensure(
    lambda result, n_more_estimators: _seed_vector_valid(result, n_more_estimators),
    "seed vector must contain one sklearn-compatible int32-range seed per new estimator",
)
def bagging_fit_seeds(
    n_more_estimators: int,
    previous_estimators: int,
    *,
    random_state: RandomStateLike = None,
) -> NDArray[np.int64]:
    from sklearn.utils import check_random_state
    """Generate sklearn bagging's per-estimator seeds after warm-start advancement."""
    rng = check_random_state(random_state)
    if previous_estimators > 0:
        rng.randint(MAX_INT, size=previous_estimators)
    return np.asarray(rng.randint(MAX_INT, size=n_more_estimators), dtype=np.int64)
