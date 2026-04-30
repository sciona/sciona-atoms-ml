"""Ghost witnesses for BIRCH node-buffer helpers."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def _check_centroid_matrix(active_centroids: AbstractArray) -> tuple[int, int]:
    if len(active_centroids.shape) != 2:
        raise ValueError("active_centroids must be 2D")
    rows = int(active_centroids.shape[0])
    cols = int(active_centroids.shape[1])
    if rows < 0 or cols < 1:
        raise ValueError("active_centroids must have nonnegative rows and at least one feature")
    return rows, cols


def _check_centroid_vector(candidate_centroid: AbstractArray, width: int) -> None:
    if len(candidate_centroid.shape) != 1 or int(candidate_centroid.shape[0]) != width:
        raise ValueError("candidate_centroid must match centroid width")


def _check_squared_norms(active_squared_norms: AbstractArray, expected_length: int) -> None:
    if len(active_squared_norms.shape) != 1 or int(active_squared_norms.shape[0]) != expected_length:
        raise ValueError("active_squared_norms must be 1D and match centroid row count")


def witness_birch_append_active_count(current_count: int) -> int:
    """Describe the active subcluster count after one append."""
    if current_count < 0:
        raise ValueError("current_count must be nonnegative")
    return current_count + 1


def witness_birch_append_centroids(
    active_centroids: AbstractArray,
    candidate_centroid: AbstractArray,
) -> AbstractArray:
    """Describe centroid rows after one append."""
    rows, cols = _check_centroid_matrix(active_centroids)
    _check_centroid_vector(candidate_centroid, cols)
    return AbstractArray(shape=(rows + 1, cols), dtype="float64")


def witness_birch_append_squared_norms(
    active_squared_norms: AbstractArray,
    candidate_sq_norm: float,
) -> AbstractArray:
    """Describe squared norms after one append."""
    del candidate_sq_norm
    if len(active_squared_norms.shape) != 1:
        raise ValueError("active_squared_norms must be 1D")
    length = int(active_squared_norms.shape[0])
    return AbstractArray(shape=(length + 1,), dtype="float64", min_val=0.0)


def witness_birch_update_split_centroids(
    active_centroids: AbstractArray,
    replace_index: int,
    replacement_centroid: AbstractArray,
    appended_centroid: AbstractArray,
) -> AbstractArray:
    """Describe centroid rows after one split replacement and append."""
    rows, cols = _check_centroid_matrix(active_centroids)
    if not 0 <= replace_index < rows:
        raise ValueError("replace_index must refer to an existing centroid row")
    _check_centroid_vector(replacement_centroid, cols)
    _check_centroid_vector(appended_centroid, cols)
    return AbstractArray(shape=(rows + 1, cols), dtype="float64")


def witness_birch_update_split_squared_norms(
    active_squared_norms: AbstractArray,
    replace_index: int,
    replacement_sq_norm: float,
    appended_sq_norm: float,
) -> AbstractArray:
    """Describe squared norms after one split replacement and append."""
    del replacement_sq_norm, appended_sq_norm
    if len(active_squared_norms.shape) != 1:
        raise ValueError("active_squared_norms must be 1D")
    length = int(active_squared_norms.shape[0])
    if not 0 <= replace_index < length:
        raise ValueError("replace_index must refer to an existing squared norm")
    return AbstractArray(shape=(length + 1,), dtype="float64", min_val=0.0)
