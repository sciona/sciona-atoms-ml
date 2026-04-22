"""Deterministic IsolationForest helper atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_isolation_forest_average_path_length,
    witness_isolation_forest_leaf_depths,
    witness_isolation_forest_raw_scores,
)


def _sample_counts_valid(n_samples_leaf: NDArray[np.int64]) -> bool:
    try:
        values = np.asarray(n_samples_leaf)
    except (TypeError, ValueError):
        return False
    return bool(
        values.ndim == 1
        and values.shape[0] >= 1
        and np.issubdtype(values.dtype, np.integer)
        and np.all(values >= 0)
    )


def _finite_nonnegative_vector(values: NDArray[np.float64]) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 1 and array.shape[0] >= 1 and np.all(np.isfinite(array)) and np.all(array >= 0.0))


def _leaf_indices_valid(
    leaf_indices: NDArray[np.int64],
    tree_decision_path_lengths: NDArray[np.float64],
    tree_average_path_lengths: NDArray[np.float64],
) -> bool:
    try:
        indices = np.asarray(leaf_indices)
        path_lengths = np.asarray(tree_decision_path_lengths, dtype=np.float64)
        average_lengths = np.asarray(tree_average_path_lengths, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(
        indices.ndim == 1
        and indices.shape[0] >= 1
        and np.issubdtype(indices.dtype, np.integer)
        and path_lengths.ndim == 1
        and average_lengths.ndim == 1
        and path_lengths.shape == average_lengths.shape
        and path_lengths.shape[0] >= 1
        and np.all(indices >= 0)
        and np.all(indices < path_lengths.shape[0])
        and np.all(np.isfinite(path_lengths))
        and np.all(np.isfinite(average_lengths))
        and np.all(path_lengths >= 0.0)
        and np.all(average_lengths >= 0.0)
    )


def _positive_int(value: int) -> bool:
    return bool(isinstance(value, int) and not isinstance(value, bool) and value >= 1)


def _nonnegative_int(value: int) -> bool:
    return bool(isinstance(value, int) and not isinstance(value, bool) and value >= 0)


def _path_lengths_valid(result: NDArray[np.float64], n_samples_leaf: NDArray[np.int64]) -> bool:
    values = np.asarray(result, dtype=np.float64)
    counts = np.asarray(n_samples_leaf)
    return bool(
        values.shape == counts.shape
        and np.all(np.isfinite(values))
        and np.all(values >= 0.0)
        and np.all(values[counts <= 1] == 0.0)
        and np.all(values[counts == 2] == 1.0)
    )


def _leaf_depths_valid(result: NDArray[np.float64], leaf_indices: NDArray[np.int64]) -> bool:
    values = np.asarray(result, dtype=np.float64)
    indices = np.asarray(leaf_indices)
    return bool(values.shape == indices.shape and np.all(np.isfinite(values)) and np.all(values >= 0.0))


def _scores_valid(result: NDArray[np.float64], depths: NDArray[np.float64]) -> bool:
    values = np.asarray(result, dtype=np.float64)
    input_values = np.asarray(depths, dtype=np.float64)
    return bool(
        values.shape == input_values.shape
        and np.all(np.isfinite(values))
        and np.all(values > 0.0)
        and np.all(values <= 1.0)
    )


@register_atom(witness_isolation_forest_average_path_length)
@icontract.require(lambda n_samples_leaf: _sample_counts_valid(n_samples_leaf), "n_samples_leaf must be a nonempty integer vector")
@icontract.ensure(lambda result, n_samples_leaf: _path_lengths_valid(result, n_samples_leaf), "path lengths must follow sklearn's leaf-count cases")
def isolation_forest_average_path_length(n_samples_leaf: NDArray[np.int64]) -> NDArray[np.float64]:
    """Compute sklearn's average path length for isolation-tree leaf counts."""
    counts = np.asarray(n_samples_leaf, dtype=np.int64)
    flat_counts = counts.reshape((1, -1)).astype(np.float64, copy=False)
    average_path_length = np.zeros(flat_counts.shape, dtype=np.float64)

    mask_1 = flat_counts <= 1.0
    mask_2 = flat_counts == 2.0
    not_mask = ~np.logical_or(mask_1, mask_2)

    average_path_length[mask_1] = 0.0
    average_path_length[mask_2] = 1.0
    average_path_length[not_mask] = (
        2.0 * (np.log(flat_counts[not_mask] - 1.0) + np.euler_gamma)
        - 2.0 * (flat_counts[not_mask] - 1.0) / flat_counts[not_mask]
    )
    return np.asarray(average_path_length.reshape(counts.shape), dtype=np.float64)


@register_atom(witness_isolation_forest_leaf_depths)
@icontract.require(
    lambda leaf_indices, tree_decision_path_lengths, tree_average_path_lengths: _leaf_indices_valid(
        leaf_indices,
        tree_decision_path_lengths,
        tree_average_path_lengths,
    ),
    "leaf_indices must index matching path-length vectors",
)
@icontract.ensure(lambda result, leaf_indices: _leaf_depths_valid(result, leaf_indices), "leaf depths must be finite and nonnegative")
def isolation_forest_leaf_depths(
    leaf_indices: NDArray[np.int64],
    tree_decision_path_lengths: NDArray[np.float64],
    tree_average_path_lengths: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute one tree's depth contribution for samples that reached leaves."""
    indices = np.asarray(leaf_indices, dtype=np.int64)
    decision_lengths = np.asarray(tree_decision_path_lengths, dtype=np.float64)
    average_lengths = np.asarray(tree_average_path_lengths, dtype=np.float64)
    return np.asarray(decision_lengths[indices] + average_lengths[indices] - 1.0, dtype=np.float64)


@register_atom(witness_isolation_forest_raw_scores)
@icontract.require(lambda depths: _finite_nonnegative_vector(depths), "depths must be a finite nonnegative vector")
@icontract.require(lambda n_estimators: _positive_int(n_estimators), "n_estimators must be a positive integer")
@icontract.require(lambda max_samples: _nonnegative_int(max_samples), "max_samples must be a nonnegative integer")
@icontract.ensure(lambda result, depths: _scores_valid(result, depths), "raw scores must stay in the open-closed unit interval")
def isolation_forest_raw_scores(
    depths: NDArray[np.float64],
    *,
    n_estimators: int,
    max_samples: int,
) -> NDArray[np.float64]:
    """Convert accumulated tree depths to sklearn's positive raw anomaly scores."""
    depth_values = np.asarray(depths, dtype=np.float64)
    average_path_length_max_samples = isolation_forest_average_path_length(np.asarray([max_samples], dtype=np.int64))
    denominator = int(n_estimators) * average_path_length_max_samples
    exponents = -np.divide(
        depth_values,
        denominator,
        out=np.ones_like(depth_values),
        where=denominator != 0,
    )
    return np.asarray(2.0**exponents, dtype=np.float64)
