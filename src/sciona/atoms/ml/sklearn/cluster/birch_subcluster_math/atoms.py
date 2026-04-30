"""BIRCH subcluster statistics and merge helpers adapted from scikit-learn."""

from __future__ import annotations

import math

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .state_models import BirchSubclusterStats
from .witnesses import (
    witness_birch_subcluster_merge,
    witness_birch_subcluster_radius,
    witness_birch_subcluster_singleton,
    witness_birch_subcluster_squared_radius,
    witness_birch_subcluster_update,
)


def _finite_vector(value: object) -> bool:
    try:
        vector = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(vector.ndim == 1 and vector.shape[0] >= 1 and np.all(np.isfinite(vector)))


def _positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


def _nonnegative_finite_float(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(float(value)) and float(value) >= 0.0


def _positive_finite_float(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(float(value)) and float(value) > 0.0


def _state_valid(value: object) -> bool:
    if not isinstance(value, BirchSubclusterStats):
        return False
    if not _positive_int(value.n_samples):
        return False
    if not _finite_vector(value.linear_sum) or not _finite_vector(value.centroid):
        return False
    linear_sum = np.asarray(value.linear_sum, dtype=np.float64)
    centroid = np.asarray(value.centroid, dtype=np.float64)
    if linear_sum.shape != centroid.shape:
        return False
    if not _nonnegative_finite_float(value.squared_sum) or not _nonnegative_finite_float(value.sq_norm):
        return False
    return bool(np.allclose(centroid, linear_sum / value.n_samples) and np.isclose(value.sq_norm, float(np.dot(centroid, centroid))))


def _same_width(left: BirchSubclusterStats, right: BirchSubclusterStats) -> bool:
    return bool(_state_valid(left) and _state_valid(right) and left.linear_sum.shape == right.linear_sum.shape)


def _merge_result_valid(value: object, base_state: BirchSubclusterStats) -> bool:
    return bool(
        isinstance(value, tuple)
        and len(value) == 2
        and isinstance(value[0], bool)
        and isinstance(value[1], BirchSubclusterStats)
        and _state_valid(value[1])
        and value[1].linear_sum.shape == base_state.linear_sum.shape
    )


@register_atom(witness_birch_subcluster_singleton)
@icontract.require(lambda linear_sum: _finite_vector(linear_sum), "linear_sum must be a nonempty finite 1D vector")
@icontract.ensure(lambda result: _state_valid(result), "result must be valid singleton BIRCH subcluster statistics")
def birch_subcluster_singleton(linear_sum: NDArray[np.float64]) -> BirchSubclusterStats:
    """Construct singleton BIRCH subcluster statistics from one feature vector."""
    linear_sum_values = np.asarray(linear_sum, dtype=np.float64)
    squared_sum = float(np.dot(linear_sum_values, linear_sum_values))
    return BirchSubclusterStats(
        n_samples=1,
        linear_sum=linear_sum_values.copy(),
        squared_sum=squared_sum,
        centroid=linear_sum_values.copy(),
        sq_norm=squared_sum,
    )


@register_atom(witness_birch_subcluster_update)
@icontract.require(lambda base_state: _state_valid(base_state), "base_state must be valid BIRCH subcluster statistics")
@icontract.require(lambda added_state: _state_valid(added_state), "added_state must be valid BIRCH subcluster statistics")
@icontract.require(lambda base_state, added_state: _same_width(base_state, added_state), "base_state and added_state must share the same feature width")
@icontract.ensure(lambda result: _state_valid(result), "result must be valid updated BIRCH subcluster statistics")
def birch_subcluster_update(
    base_state: BirchSubclusterStats,
    added_state: BirchSubclusterStats,
) -> BirchSubclusterStats:
    """Accumulate one BIRCH subcluster into another using sklearn's update rule."""
    new_n_samples = base_state.n_samples + added_state.n_samples
    new_linear_sum = np.asarray(base_state.linear_sum + added_state.linear_sum, dtype=np.float64)
    new_squared_sum = float(base_state.squared_sum + added_state.squared_sum)
    new_centroid = np.asarray(new_linear_sum / new_n_samples, dtype=np.float64)
    new_sq_norm = float(np.dot(new_centroid, new_centroid))
    return BirchSubclusterStats(
        n_samples=new_n_samples,
        linear_sum=new_linear_sum,
        squared_sum=new_squared_sum,
        centroid=new_centroid,
        sq_norm=new_sq_norm,
    )


@register_atom(witness_birch_subcluster_squared_radius)
@icontract.require(lambda state: _state_valid(state), "state must be valid BIRCH subcluster statistics")
@icontract.ensure(lambda result: _nonnegative_finite_float(result) or np.isfinite(float(result)), "squared radius must be finite")
def birch_subcluster_squared_radius(state: BirchSubclusterStats) -> float:
    """Compute sklearn's BIRCH subcluster squared-radius expression."""
    return float(state.squared_sum / state.n_samples - state.sq_norm)


@register_atom(witness_birch_subcluster_merge)
@icontract.require(lambda base_state: _state_valid(base_state), "base_state must be valid BIRCH subcluster statistics")
@icontract.require(lambda nominee_state: _state_valid(nominee_state), "nominee_state must be valid BIRCH subcluster statistics")
@icontract.require(lambda base_state, nominee_state: _same_width(base_state, nominee_state), "base_state and nominee_state must share the same feature width")
@icontract.require(lambda threshold: _positive_finite_float(threshold), "threshold must be positive")
@icontract.ensure(lambda result, base_state: _merge_result_valid(result, base_state), "merge result must contain a Boolean flag and valid BIRCH subcluster statistics")
def birch_subcluster_merge(
    base_state: BirchSubclusterStats,
    nominee_state: BirchSubclusterStats,
    threshold: float,
) -> tuple[bool, BirchSubclusterStats]:
    """Attempt sklearn's thresholded BIRCH subcluster merge and return the resulting state."""
    merged_state = birch_subcluster_update(base_state, nominee_state)
    merged_sq_radius = birch_subcluster_squared_radius(merged_state)
    if merged_sq_radius <= float(threshold) ** 2:
        return True, merged_state
    return False, base_state


@register_atom(witness_birch_subcluster_radius)
@icontract.require(lambda state: _state_valid(state), "state must be valid BIRCH subcluster statistics")
@icontract.ensure(lambda result: _nonnegative_finite_float(result), "radius must be a finite nonnegative float")
def birch_subcluster_radius(state: BirchSubclusterStats) -> float:
    """Compute sklearn's public BIRCH subcluster radius property."""
    return float(math.sqrt(max(0.0, birch_subcluster_squared_radius(state))))
