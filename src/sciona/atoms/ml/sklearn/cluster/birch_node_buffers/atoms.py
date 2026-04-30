"""BIRCH node-buffer helpers adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_birch_append_active_count,
    witness_birch_append_centroids,
    witness_birch_append_squared_norms,
    witness_birch_update_split_centroids,
    witness_birch_update_split_squared_norms,
)


def _nonnegative_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _finite_centroid_matrix(value: object) -> bool:
    try:
        matrix = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(matrix.ndim == 2 and matrix.shape[1] >= 1 and np.all(np.isfinite(matrix)))


def _finite_centroid_vector(value: object, width: int | None = None) -> bool:
    try:
        vector = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    if not bool(vector.ndim == 1 and np.all(np.isfinite(vector))):
        return False
    if width is None:
        return True
    return bool(vector.shape[0] == width)


def _finite_nonnegative_vector(value: object, length: int | None = None) -> bool:
    try:
        vector = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    if not bool(vector.ndim == 1 and np.all(np.isfinite(vector)) and np.all(vector >= 0.0)):
        return False
    if length is None:
        return True
    return bool(vector.shape[0] == length)


def _nonnegative_float(value: object) -> bool:
    return bool(isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(float(value)) and float(value) >= 0.0)


def _replace_index_valid(replace_index: object, active_centroids: NDArray[np.float64]) -> bool:
    return bool(
        isinstance(replace_index, int)
        and not isinstance(replace_index, bool)
        and 0 <= replace_index < np.asarray(active_centroids).shape[0]
    )


def _appended_centroids_valid(result: object, active_centroids: NDArray[np.float64]) -> bool:
    values = np.asarray(result, dtype=np.float64)
    prior = np.asarray(active_centroids, dtype=np.float64)
    return bool(
        values.ndim == 2
        and values.shape == (prior.shape[0] + 1, prior.shape[1])
        and np.all(np.isfinite(values))
    )


def _appended_squared_norms_valid(result: object, active_squared_norms: NDArray[np.float64]) -> bool:
    values = np.asarray(result, dtype=np.float64)
    prior = np.asarray(active_squared_norms, dtype=np.float64)
    return bool(
        values.ndim == 1
        and values.shape == (prior.shape[0] + 1,)
        and np.all(np.isfinite(values))
        and np.all(values >= 0.0)
    )


@register_atom(witness_birch_append_active_count)
@icontract.require(lambda current_count: _nonnegative_int(current_count), "current_count must be a nonnegative integer")
@icontract.ensure(lambda result, current_count: result == current_count + 1, "result must increment the active subcluster count by one")
def birch_append_active_count(current_count: int) -> int:
    """Increment the active BIRCH subcluster count for one append."""
    return current_count + 1


@register_atom(witness_birch_append_centroids)
@icontract.require(lambda active_centroids: _finite_centroid_matrix(active_centroids), "active_centroids must be a finite 2D matrix")
@icontract.require(lambda candidate_centroid, active_centroids: _finite_centroid_vector(candidate_centroid, np.asarray(active_centroids).shape[1]), "candidate_centroid must be a finite 1D vector matching centroid width")
@icontract.ensure(lambda result, active_centroids: _appended_centroids_valid(result, active_centroids), "result must append one centroid row")
def birch_append_centroids(
    active_centroids: NDArray[np.float64],
    candidate_centroid: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Append one centroid row the way sklearn updates a BIRCH node view."""
    return np.asarray(
        np.vstack(
            (
                np.asarray(active_centroids, dtype=np.float64),
                np.asarray(candidate_centroid, dtype=np.float64).reshape(1, -1),
            )
        ),
        dtype=np.float64,
    )


@register_atom(witness_birch_append_squared_norms)
@icontract.require(lambda active_squared_norms: _finite_nonnegative_vector(active_squared_norms), "active_squared_norms must be a finite nonnegative 1D vector")
@icontract.require(lambda candidate_sq_norm: _nonnegative_float(candidate_sq_norm), "candidate_sq_norm must be a finite nonnegative scalar")
@icontract.ensure(lambda result, active_squared_norms: _appended_squared_norms_valid(result, active_squared_norms), "result must append one squared norm")
def birch_append_squared_norms(
    active_squared_norms: NDArray[np.float64],
    candidate_sq_norm: float,
) -> NDArray[np.float64]:
    """Append one squared norm the way sklearn updates a BIRCH node view."""
    return np.asarray(
        np.concatenate(
            (
                np.asarray(active_squared_norms, dtype=np.float64),
                np.asarray([candidate_sq_norm], dtype=np.float64),
            )
        ),
        dtype=np.float64,
    )


@register_atom(witness_birch_update_split_centroids)
@icontract.require(lambda active_centroids: _finite_centroid_matrix(active_centroids), "active_centroids must be a finite 2D matrix")
@icontract.require(lambda replace_index, active_centroids: _replace_index_valid(replace_index, active_centroids), "replace_index must be a valid centroid row index")
@icontract.require(lambda replacement_centroid, active_centroids: _finite_centroid_vector(replacement_centroid, np.asarray(active_centroids).shape[1]), "replacement_centroid must be a finite 1D vector matching centroid width")
@icontract.require(lambda appended_centroid, active_centroids: _finite_centroid_vector(appended_centroid, np.asarray(active_centroids).shape[1]), "appended_centroid must be a finite 1D vector matching centroid width")
@icontract.ensure(lambda result, active_centroids: _appended_centroids_valid(result, active_centroids), "result must replace one row and append one centroid row")
def birch_update_split_centroids(
    active_centroids: NDArray[np.float64],
    replace_index: int,
    replacement_centroid: NDArray[np.float64],
    appended_centroid: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Replace one centroid row, then append another, like sklearn's split update."""
    updated = np.asarray(active_centroids, dtype=np.float64).copy()
    updated[replace_index] = np.asarray(replacement_centroid, dtype=np.float64)
    return birch_append_centroids(updated, appended_centroid)


@register_atom(witness_birch_update_split_squared_norms)
@icontract.require(lambda active_squared_norms: _finite_nonnegative_vector(active_squared_norms), "active_squared_norms must be a finite nonnegative 1D vector")
@icontract.require(lambda replace_index, active_squared_norms: isinstance(replace_index, int) and not isinstance(replace_index, bool) and 0 <= replace_index < np.asarray(active_squared_norms).shape[0], "replace_index must be a valid squared-norm index")
@icontract.require(lambda replacement_sq_norm: _nonnegative_float(replacement_sq_norm), "replacement_sq_norm must be a finite nonnegative scalar")
@icontract.require(lambda appended_sq_norm: _nonnegative_float(appended_sq_norm), "appended_sq_norm must be a finite nonnegative scalar")
@icontract.ensure(lambda result, active_squared_norms: _appended_squared_norms_valid(result, active_squared_norms), "result must replace one entry and append one squared norm")
def birch_update_split_squared_norms(
    active_squared_norms: NDArray[np.float64],
    replace_index: int,
    replacement_sq_norm: float,
    appended_sq_norm: float,
) -> NDArray[np.float64]:
    """Replace one squared norm, then append another, like sklearn's split update."""
    updated = np.asarray(active_squared_norms, dtype=np.float64).copy()
    updated[replace_index] = float(replacement_sq_norm)
    return birch_append_squared_norms(updated, float(appended_sq_norm))
