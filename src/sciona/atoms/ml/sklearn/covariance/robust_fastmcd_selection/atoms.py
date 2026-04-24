"""Multivariate FastMCD selection and scheduling helper atoms adapted from scikit-learn."""

from __future__ import annotations

from numbers import Integral

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_fast_mcd_best_candidate_indices,
    witness_fast_mcd_gather_best_candidates,
    witness_fast_mcd_large_sample_schedule,
    witness_fast_mcd_place_merged_results,
    witness_fast_mcd_trial_plan,
)


def _positive_int(value: int) -> bool:
    return bool(isinstance(value, int) and not isinstance(value, bool) and value >= 1)


def _finite_vector(values: object) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 1 and array.shape[0] >= 1 and np.all(np.isfinite(array)))


def _finite_matrix(values: object) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 2 and array.shape[0] >= 1 and array.shape[1] >= 1 and np.all(np.isfinite(array)))


def _finite_rank3_square(values: object) -> bool:
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


def _bool_matrix(values: object) -> bool:
    array = np.asarray(values)
    return bool(array.ndim == 2 and array.shape[0] >= 1 and array.shape[1] >= 1)


def _aligned_candidate_shapes(
    locations: object,
    covariances: object,
    supports: object,
    distances: object,
) -> bool:
    if not (_finite_matrix(locations) and _finite_rank3_square(covariances) and _bool_matrix(supports) and _finite_matrix(distances)):
        return False
    location_values = np.asarray(locations, dtype=np.float64)
    covariance_values = np.asarray(covariances, dtype=np.float64)
    support_values = np.asarray(supports)
    distance_values = np.asarray(distances, dtype=np.float64)
    return bool(
        location_values.shape[0] == covariance_values.shape[0] == support_values.shape[0] == distance_values.shape[0]
        and location_values.shape[1] == covariance_values.shape[1]
        and support_values.shape[1] == distance_values.shape[1]
    )


def _index_vector(values: object, upper_bound: int) -> bool:
    array = np.asarray(values)
    return bool(
        array.ndim == 1
        and array.shape[0] >= 1
        and np.issubdtype(array.dtype, np.integer)
        and np.all(array >= 0)
        and np.all(array < upper_bound)
        and np.unique(array).shape[0] == array.shape[0]
    )


def _trial_plan_valid(result: tuple[bool, int]) -> bool:
    return bool(isinstance(result, tuple) and len(result) == 2 and isinstance(result[0], bool) and _positive_int(result[1]))


def _best_indices_valid(result: NDArray[np.int64], determinants: NDArray[np.float64], select: int) -> bool:
    indices = np.asarray(result)
    return bool(
        indices.ndim == 1
        and indices.shape[0] == select
        and np.issubdtype(indices.dtype, np.integer)
        and np.unique(indices).shape[0] == indices.shape[0]
        and np.all(indices >= 0)
        and np.all(indices < np.asarray(determinants, dtype=np.float64).shape[0])
    )


def _gathered_candidates_valid(
    result: tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_], NDArray[np.float64]],
    locations: NDArray[np.float64],
    supports: NDArray[np.bool_],
    indices: NDArray[np.int64],
) -> bool:
    if not (isinstance(result, tuple) and len(result) == 4):
        return False
    selected = np.asarray(indices).shape[0]
    n_features = np.asarray(locations, dtype=np.float64).shape[1]
    n_samples = np.asarray(supports).shape[1]
    best_locations, best_covariances, best_supports, best_distances = result
    return bool(
        np.asarray(best_locations, dtype=np.float64).shape == (selected, n_features)
        and np.asarray(best_covariances, dtype=np.float64).shape == (selected, n_features, n_features)
        and np.asarray(best_supports).shape == (selected, n_samples)
        and np.asarray(best_distances, dtype=np.float64).shape == (selected, n_samples)
    )


def _schedule_valid(result: tuple[int, int, int, int, int, int, int, int, int, int], n_samples: int) -> bool:
    if not (isinstance(result, tuple) and len(result) == 10 and all(_positive_int(value) for value in result)):
        return False
    n_subsets, n_samples_subsets, h_subset, n_trials_tot, n_best_sub, n_trials, n_best_tot, n_samples_merged, h_merged, n_best_merged = result
    return bool(
        n_subsets >= 1
        and n_samples_subsets >= 1
        and h_subset >= 1
        and n_trials_tot == 500
        and n_best_sub >= 1
        and n_trials >= 1
        and n_best_tot == n_subsets * n_best_sub
        and n_samples_merged <= n_samples
        and h_merged >= 1
        and n_best_merged >= 1
    )


def _placed_results_valid(
    result: tuple[NDArray[np.bool_], NDArray[np.float64]],
    n_samples: int,
) -> bool:
    if not (isinstance(result, tuple) and len(result) == 2):
        return False
    support, distances = result
    return bool(
        np.asarray(support).shape == (n_samples,)
        and np.asarray(distances, dtype=np.float64).shape == (n_samples,)
        and np.all(np.isfinite(np.asarray(distances, dtype=np.float64)))
    )


def _validate_estimate_tuple(n_trials: tuple[object, object]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    if len(n_trials) != 2:
        raise ValueError("n_trials estimate tuple must contain (locations, covariances)")
    locations = np.asarray(n_trials[0], dtype=np.float64)
    covariances = np.asarray(n_trials[1], dtype=np.float64)
    if not _finite_matrix(locations):
        raise ValueError("initial locations must be a finite 2D array")
    if not _finite_rank3_square(covariances):
        raise ValueError("initial covariances must be a finite 3D square tensor")
    if locations.shape[0] != covariances.shape[0]:
        raise ValueError("initial locations and covariances must share the same trial count")
    if locations.shape[1] != covariances.shape[1]:
        raise ValueError("initial location and covariance feature dimensions must match")
    return locations, covariances


@register_atom(witness_fast_mcd_trial_plan)
@icontract.require(lambda n_trials: not isinstance(n_trials, bool), "n_trials must not be boolean")
@icontract.ensure(lambda result: _trial_plan_valid(result), "trial plan must be a (run_from_estimates, n_trials) tuple")
def fast_mcd_trial_plan(
    n_trials: int | tuple[NDArray[np.float64], NDArray[np.float64]],
) -> tuple[bool, int]:
    """Resolve whether FastMCD should start from random supports or supplied estimates."""
    if isinstance(n_trials, Integral) and not isinstance(n_trials, bool):
        if int(n_trials) < 1:
            raise ValueError("n_trials must be positive")
        return False, int(n_trials)
    if isinstance(n_trials, tuple):
        locations, _ = _validate_estimate_tuple(n_trials)
        return True, int(locations.shape[0])
    raise TypeError(
        "Invalid 'n_trials' parameter, expected tuple or  integer, got %s (%s)"
        % (n_trials, type(n_trials))
    )


@register_atom(witness_fast_mcd_best_candidate_indices)
@icontract.require(lambda determinants: _finite_vector(determinants), "determinants must be a finite nonempty vector")
@icontract.require(lambda select: _positive_int(select), "select must be a positive integer")
@icontract.require(lambda determinants, select: int(select) <= np.asarray(determinants, dtype=np.float64).shape[0], "select must not exceed the number of determinants")
@icontract.ensure(lambda result, determinants, select: _best_indices_valid(result, determinants, select), "result must be a unique integer index vector within the determinant range")
def fast_mcd_best_candidate_indices(
    determinants: NDArray[np.float64],
    *,
    select: int = 1,
) -> NDArray[np.int64]:
    """Rank candidate estimates by determinant and return sklearn's selected indices."""
    determinant_values = np.asarray(determinants, dtype=np.float64)
    return np.asarray(np.argsort(determinant_values)[:select], dtype=np.int64)


@register_atom(witness_fast_mcd_gather_best_candidates)
@icontract.require(
    lambda locations, covariances, supports, distances: _aligned_candidate_shapes(locations, covariances, supports, distances),
    "candidate tensors must be finite, nonempty, and aligned on candidate, feature, and sample axes",
)
@icontract.require(
    lambda locations, indices: _index_vector(indices, np.asarray(locations, dtype=np.float64).shape[0]),
    "indices must be a unique integer vector within the candidate range",
)
@icontract.ensure(
    lambda result, locations, supports, indices: _gathered_candidates_valid(result, locations, supports, indices),
    "gathered candidate tensors must align with the requested selection size",
)
def fast_mcd_gather_best_candidates(
    locations: NDArray[np.float64],
    covariances: NDArray[np.float64],
    supports: NDArray[np.bool_],
    distances: NDArray[np.float64],
    indices: NDArray[np.int64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_], NDArray[np.float64]]:
    """Gather the ranked FastMCD candidate tensors selected from a candidate pool."""
    gathered = np.asarray(indices, dtype=np.int64)
    return (
        np.asarray(locations, dtype=np.float64)[gathered],
        np.asarray(covariances, dtype=np.float64)[gathered],
        np.asarray(supports, dtype=np.bool_)[gathered],
        np.asarray(distances, dtype=np.float64)[gathered],
    )


@register_atom(witness_fast_mcd_large_sample_schedule)
@icontract.require(lambda n_samples: _positive_int(n_samples) and n_samples > 500, "n_samples must be a positive integer greater than 500")
@icontract.require(lambda n_features: _positive_int(n_features) and n_features > 1, "n_features must be a positive integer greater than 1")
@icontract.require(lambda n_support: _positive_int(n_support), "n_support must be a positive integer")
@icontract.require(lambda n_samples, n_support: int(n_support) <= int(n_samples), "n_support must not exceed n_samples")
@icontract.ensure(lambda result, n_samples: _schedule_valid(result, n_samples), "schedule must contain positive sklearn FastMCD large-sample constants")
def fast_mcd_large_sample_schedule(
    n_samples: int,
    n_features: int,
    n_support: int,
) -> tuple[int, int, int, int, int, int, int, int, int, int]:
    """Compute sklearn's deterministic scheduling constants for large multivariate FastMCD runs."""
    n_subsets = int(n_samples) // 300
    n_samples_subsets = int(n_samples) // n_subsets
    h_subset = int(np.ceil(n_samples_subsets * (int(n_support) / float(n_samples))))
    n_trials_tot = 500
    n_best_sub = 10
    n_trials = max(10, n_trials_tot // n_subsets)
    n_best_tot = n_subsets * n_best_sub
    n_samples_merged = min(1500, int(n_samples))
    h_merged = int(np.ceil(n_samples_merged * (int(n_support) / float(n_samples))))
    n_best_merged = 10 if int(n_samples) > 1500 else 1
    return (
        n_subsets,
        n_samples_subsets,
        h_subset,
        n_trials_tot,
        n_best_sub,
        n_trials,
        n_best_tot,
        n_samples_merged,
        h_merged,
        n_best_merged,
    )


@register_atom(witness_fast_mcd_place_merged_results)
@icontract.require(lambda n_samples: _positive_int(n_samples), "n_samples must be a positive integer")
@icontract.require(lambda selection: _index_vector(selection, int(np.max(np.asarray(selection))) + 1) if np.asarray(selection).size else False, "selection must be a unique integer vector")
@icontract.require(lambda selection, merged_support: np.asarray(selection).ndim == 1 and np.asarray(merged_support).ndim == 1 and np.asarray(selection).shape[0] == np.asarray(merged_support).shape[0], "selection and merged_support must be aligned vectors")
@icontract.require(lambda selection, merged_distances: np.asarray(selection).ndim == 1 and _finite_vector(merged_distances) and np.asarray(selection).shape[0] == np.asarray(merged_distances, dtype=np.float64).shape[0], "selection and merged_distances must be aligned vectors")
@icontract.require(lambda n_samples, selection: np.asarray(selection).size >= 1 and np.all(np.asarray(selection) < int(n_samples)), "selection indices must fall within the sample range")
@icontract.ensure(lambda result, n_samples: _placed_results_valid(result, n_samples), "result must be full-length support and distance vectors")
def fast_mcd_place_merged_results(
    n_samples: int,
    selection: NDArray[np.int64],
    merged_support: NDArray[np.bool_],
    merged_distances: NDArray[np.float64],
) -> tuple[NDArray[np.bool_], NDArray[np.float64]]:
    """Scatter merged-set FastMCD support and distances back onto the full dataset indices."""
    support = np.zeros(int(n_samples), dtype=np.bool_)
    distances = np.zeros(int(n_samples), dtype=np.float64)
    selected = np.asarray(selection, dtype=np.int64)
    support[selected] = np.asarray(merged_support, dtype=np.bool_)
    distances[selected] = np.asarray(merged_distances, dtype=np.float64)
    return support, distances
