"""Deterministic FastMCD candidate-pool helper atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray
from sklearn.utils import check_random_state

from sciona.atoms.ml.sklearn.covariance.robust_fastmcd_c_step import fast_mcd_c_step
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_fast_mcd_candidate_pool_from_estimates,
    witness_fast_mcd_candidate_pool_from_random_starts,
)

RandomStateLike = int | np.random.RandomState | None
CandidatePool = tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_], NDArray[np.float64]]


def _finite_matrix(values: object) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 2 and array.shape[0] >= 2 and array.shape[1] >= 1 and np.all(np.isfinite(array)))


def _finite_covariance_stack(values: object) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(
        array.ndim == 3
        and array.shape[0] >= 1
        and array.shape[1] >= 1
        and array.shape[1] == array.shape[2]
        and np.all(np.isfinite(array))
    )


def _positive_int(value: int) -> bool:
    return bool(isinstance(value, int) and not isinstance(value, bool) and value >= 1)


def _n_iter_valid(value: int) -> bool:
    return bool(isinstance(value, int) and not isinstance(value, bool) and value >= 0)


def _support_size_valid(n_support: int, X: NDArray[np.float64]) -> bool:
    return bool(_positive_int(n_support) and n_support <= np.asarray(X, dtype=np.float64).shape[0])


def _estimate_stacks_valid(X: NDArray[np.float64], initial_locations: NDArray[np.float64], initial_covariances: NDArray[np.float64], n_support: int) -> bool:
    X_values = np.asarray(X, dtype=np.float64)
    loc_values = np.asarray(initial_locations, dtype=np.float64)
    cov_values = np.asarray(initial_covariances, dtype=np.float64)
    return bool(
        _finite_matrix(X)
        and _finite_matrix(initial_locations)
        and _finite_covariance_stack(initial_covariances)
        and loc_values.shape[0] == cov_values.shape[0]
        and loc_values.shape[1] == X_values.shape[1]
        and cov_values.shape[1:] == (X_values.shape[1], X_values.shape[1])
        and _support_size_valid(n_support, X_values)
    )


def _candidate_pool_valid(result: CandidatePool, X: NDArray[np.float64], trial_count: int) -> bool:
    if not (isinstance(result, tuple) and len(result) == 5):
        return False
    locations, covariances, determinants, supports, distances = result
    X_values = np.asarray(X, dtype=np.float64)
    n_samples, n_features = X_values.shape
    return bool(
        np.asarray(locations, dtype=np.float64).shape == (trial_count, n_features)
        and np.asarray(covariances, dtype=np.float64).shape == (trial_count, n_features, n_features)
        and np.asarray(determinants, dtype=np.float64).shape == (trial_count,)
        and np.asarray(supports).shape == (trial_count, n_samples)
        and np.asarray(supports).dtype == np.bool_
        and np.asarray(distances, dtype=np.float64).shape == (trial_count, n_samples)
        and np.all(np.isfinite(np.asarray(distances, dtype=np.float64)))
        and np.all(np.asarray(distances, dtype=np.float64) >= 0.0)
    )


def _stack_estimates(estimates: list[tuple[NDArray[np.float64], NDArray[np.float64], float, NDArray[np.bool_], NDArray[np.float64]]]) -> CandidatePool:
    locations, covariances, determinants, supports, distances = zip(*estimates)
    return (
        np.asarray(locations, dtype=np.float64),
        np.asarray(covariances, dtype=np.float64),
        np.asarray(determinants, dtype=np.float64),
        np.asarray(supports, dtype=np.bool_),
        np.asarray(distances, dtype=np.float64),
    )


@register_atom(witness_fast_mcd_candidate_pool_from_random_starts)
@icontract.require(lambda X: _finite_matrix(X), "X must be a finite 2D sample matrix")
@icontract.require(lambda X, n_support: _support_size_valid(n_support, X), "n_support must lie in [1, n_samples]")
@icontract.require(lambda n_trials: _positive_int(n_trials), "n_trials must be a positive integer")
@icontract.require(lambda n_iter: _n_iter_valid(n_iter), "n_iter must be a nonnegative integer")
@icontract.ensure(lambda result, X, n_trials: _candidate_pool_valid(result, X, n_trials), "candidate pool tensors must align with the requested trial count")
def fast_mcd_candidate_pool_from_random_starts(
    X: NDArray[np.float64],
    n_support: int,
    n_trials: int,
    *,
    n_iter: int = 30,
    random_state: RandomStateLike = None,
) -> CandidatePool:
    """Run FastMCD c-step from random starts to build a candidate pool."""
    rng = check_random_state(random_state)
    estimates = [
        fast_mcd_c_step(
            np.asarray(X, dtype=np.float64),
            int(n_support),
            random_state=rng,
            remaining_iterations=int(n_iter),
        )
        for _ in range(int(n_trials))
    ]
    return _stack_estimates(estimates)


@register_atom(witness_fast_mcd_candidate_pool_from_estimates)
@icontract.require(lambda X, initial_locations, initial_covariances, n_support: _estimate_stacks_valid(X, initial_locations, initial_covariances, n_support), "X and initial estimate stacks must align on trial and feature axes")
@icontract.require(lambda n_iter: _n_iter_valid(n_iter), "n_iter must be a nonnegative integer")
@icontract.ensure(lambda result, X, initial_locations: _candidate_pool_valid(result, X, np.asarray(initial_locations, dtype=np.float64).shape[0]), "candidate pool tensors must align with the supplied estimate count")
def fast_mcd_candidate_pool_from_estimates(
    X: NDArray[np.float64],
    initial_locations: NDArray[np.float64],
    initial_covariances: NDArray[np.float64],
    n_support: int,
    *,
    n_iter: int = 30,
    random_state: RandomStateLike = None,
) -> CandidatePool:
    """Run FastMCD c-step from supplied estimate stacks to build a candidate pool."""
    X_values = np.asarray(X, dtype=np.float64)
    locations = np.asarray(initial_locations, dtype=np.float64)
    covariances = np.asarray(initial_covariances, dtype=np.float64)
    estimates = [
        fast_mcd_c_step(
            X_values,
            int(n_support),
            random_state=random_state,
            remaining_iterations=int(n_iter),
            initial_location=locations[j],
            initial_covariance=covariances[j],
        )
        for j in range(locations.shape[0])
    ]
    return _stack_estimates(estimates)
