"""Spectral biclustering structure atoms adapted from scikit-learn."""

from __future__ import annotations

from math import ceil, log2

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_bicluster_effective_svd_dims,
    witness_bicluster_indicator_grid,
    witness_bicluster_resolve_cluster_counts,
    witness_cocluster_indicator_matrix,
    witness_cocluster_singular_vector_count,
    witness_cocluster_split_labels,
    witness_cocluster_stacked_embedding,
)

ClusterCounts = tuple[int, int]
SplitLabels = tuple[NDArray[np.int64], NDArray[np.int64]]
BiclusterIndicatorGrid = tuple[NDArray[np.bool_], NDArray[np.bool_]]
EffectiveSvdDims = tuple[int, int]


def _positive_int(value: int) -> bool:
    return bool(isinstance(value, int) and not isinstance(value, bool) and value >= 1)


def _finite_vector(values: object) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 1 and array.shape[0] >= 1 and np.all(np.isfinite(array)))


def _nonnegative_int_vector(values: object) -> bool:
    try:
        array = np.asarray(values, dtype=np.int64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 1 and array.shape[0] >= 1)


def _finite_matrix(values: object) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 2 and array.shape[0] >= 1 and array.shape[1] >= 1 and np.all(np.isfinite(array)))


def _compatible_embedding_inputs(
    row_diag: NDArray[np.float64],
    u: NDArray[np.float64],
    col_diag: NDArray[np.float64],
    v: NDArray[np.float64],
) -> bool:
    if not (_finite_vector(row_diag) and _finite_matrix(u) and _finite_vector(col_diag) and _finite_matrix(v)):
        return False
    row_scale = np.asarray(row_diag, dtype=np.float64)
    left = np.asarray(u, dtype=np.float64)
    col_scale = np.asarray(col_diag, dtype=np.float64)
    right = np.asarray(v, dtype=np.float64)
    return bool(
        row_scale.shape[0] == left.shape[0]
        and col_scale.shape[0] == right.shape[0]
        and left.shape[1] == right.shape[1]
    )


def _stacked_embedding_valid(
    result: NDArray[np.float64],
    row_diag: NDArray[np.float64],
    col_diag: NDArray[np.float64],
    u: NDArray[np.float64],
) -> bool:
    values = np.asarray(result, dtype=np.float64)
    return bool(
        values.shape == (
            np.asarray(row_diag, dtype=np.float64).shape[0] + np.asarray(col_diag, dtype=np.float64).shape[0],
            np.asarray(u, dtype=np.float64).shape[1],
        )
        and np.all(np.isfinite(values))
    )


def _split_labels_valid(result: SplitLabels, labels: NDArray[np.int64], n_rows: int) -> bool:
    if not isinstance(result, tuple) or len(result) != 2:
        return False
    row_labels, column_labels = result
    stacked = np.asarray(labels, dtype=np.int64)
    rows = np.asarray(row_labels, dtype=np.int64)
    cols = np.asarray(column_labels, dtype=np.int64)
    return bool(
        rows.shape == (n_rows,)
        and cols.shape == (stacked.shape[0] - n_rows,)
        and np.array_equal(rows, stacked[:n_rows])
        and np.array_equal(cols, stacked[n_rows:])
    )


def _indicator_matrix_valid(result: NDArray[np.bool_], labels: NDArray[np.int64], n_clusters: int) -> bool:
    values = np.asarray(result)
    label_values = np.asarray(labels, dtype=np.int64)
    return bool(
        values.shape == (n_clusters, label_values.shape[0])
        and values.dtype == np.bool_
    )


def _method_valid(method: str) -> bool:
    return method in {"bistochastic", "scale", "log"}


def _cluster_counts_valid(value: int | tuple[int, int]) -> bool:
    if _positive_int(value):  # type: ignore[arg-type]
        return True
    if isinstance(value, tuple) and len(value) == 2:
        return all(_positive_int(item) for item in value)
    return False


def _counts_pair_valid(result: ClusterCounts) -> bool:
    return bool(isinstance(result, tuple) and len(result) == 2 and all(_positive_int(item) for item in result))


def _indicator_grid_inputs_valid(
    row_labels: NDArray[np.int64],
    column_labels: NDArray[np.int64],
    n_row_clusters: int,
    n_col_clusters: int,
) -> bool:
    return bool(
        _nonnegative_int_vector(row_labels)
        and _nonnegative_int_vector(column_labels)
        and _positive_int(n_row_clusters)
        and _positive_int(n_col_clusters)
    )


def _indicator_grid_valid(
    result: BiclusterIndicatorGrid,
    row_labels: NDArray[np.int64],
    column_labels: NDArray[np.int64],
    n_row_clusters: int,
    n_col_clusters: int,
) -> bool:
    if not isinstance(result, tuple) or len(result) != 2:
        return False
    rows, columns = result
    row_indicators = np.asarray(rows)
    column_indicators = np.asarray(columns)
    total = n_row_clusters * n_col_clusters
    return bool(
        row_indicators.shape == (total, np.asarray(row_labels, dtype=np.int64).shape[0])
        and column_indicators.shape == (total, np.asarray(column_labels, dtype=np.int64).shape[0])
        and row_indicators.dtype == np.bool_
        and column_indicators.dtype == np.bool_
    )


@register_atom(witness_cocluster_singular_vector_count)
@icontract.require(lambda n_clusters: _positive_int(n_clusters), "n_clusters must be a positive integer")
@icontract.ensure(lambda result: _positive_int(result), "singular-vector count must be a positive integer")
def cocluster_singular_vector_count(n_clusters: int) -> int:
    """Compute the singular-vector count requested by spectral coclustering."""
    return int(1 + ceil(log2(n_clusters)))


@register_atom(witness_cocluster_stacked_embedding)
@icontract.require(
    lambda row_diag, u, col_diag, v: _compatible_embedding_inputs(row_diag, u, col_diag, v),
    "row/column scales and singular vectors must be finite and shape-compatible",
)
@icontract.ensure(
    lambda result, row_diag, col_diag, u: _stacked_embedding_valid(result, row_diag, col_diag, u),
    "stacked embedding must be finite and have the concatenated sample count",
)
def cocluster_stacked_embedding(
    row_diag: NDArray[np.float64],
    u: NDArray[np.float64],
    col_diag: NDArray[np.float64],
    v: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Stack scaled left and right singular vectors for coclustering label assignment."""
    row_scale = np.asarray(row_diag, dtype=np.float64)
    left = np.asarray(u, dtype=np.float64)
    col_scale = np.asarray(col_diag, dtype=np.float64)
    right = np.asarray(v, dtype=np.float64)
    return np.asarray(
        np.vstack((row_scale[:, np.newaxis] * left, col_scale[:, np.newaxis] * right)),
        dtype=np.float64,
    )


@register_atom(witness_cocluster_split_labels)
@icontract.require(lambda labels: _nonnegative_int_vector(labels), "labels must be a nonempty integer vector")
@icontract.require(lambda labels, n_rows: _positive_int(n_rows) and int(n_rows) < np.asarray(labels, dtype=np.int64).shape[0], "n_rows must lie within the stacked label vector")
@icontract.ensure(lambda result, labels, n_rows: _split_labels_valid(result, labels, n_rows), "split labels must partition the stacked label vector")
def cocluster_split_labels(labels: NDArray[np.int64], n_rows: int) -> SplitLabels:
    """Split stacked coclustering labels into row and column partitions."""
    stacked = np.asarray(labels, dtype=np.int64)
    return (
        np.asarray(stacked[:n_rows], dtype=np.int64),
        np.asarray(stacked[n_rows:], dtype=np.int64),
    )


@register_atom(witness_cocluster_indicator_matrix)
@icontract.require(lambda labels: _nonnegative_int_vector(labels), "labels must be a nonempty integer vector")
@icontract.require(lambda n_clusters: _positive_int(n_clusters), "n_clusters must be a positive integer")
@icontract.ensure(lambda result, labels, n_clusters: _indicator_matrix_valid(result, labels, n_clusters), "indicator matrix must be boolean with one row per cluster")
def cocluster_indicator_matrix(labels: NDArray[np.int64], n_clusters: int) -> NDArray[np.bool_]:
    """Build the boolean cluster-membership matrix for one coclustering partition."""
    label_values = np.asarray(labels, dtype=np.int64)
    return np.asarray(
        np.vstack([label_values == cluster for cluster in range(n_clusters)]),
        dtype=np.bool_,
    )


@register_atom(witness_bicluster_effective_svd_dims)
@icontract.require(lambda method: isinstance(method, str) and _method_valid(method), "method must be one of 'bistochastic', 'scale', or 'log'")
@icontract.require(lambda n_components: _positive_int(n_components), "n_components must be a positive integer")
@icontract.ensure(lambda result: isinstance(result, tuple) and len(result) == 2 and all(_positive_int(v) or v == 0 for v in result), "effective SVD dimensions must be a pair of nonnegative integers")
def bicluster_effective_svd_dims(method: str, n_components: int) -> EffectiveSvdDims:
    """Resolve the requested and discarded singular-vector counts for spectral biclustering."""
    if method == "log":
        return int(n_components), 0
    return int(n_components + 1), 1


@register_atom(witness_bicluster_resolve_cluster_counts)
@icontract.require(lambda n_clusters: _cluster_counts_valid(n_clusters), "n_clusters must be a positive integer or a pair of positive integers")
@icontract.ensure(lambda result: _counts_pair_valid(result), "resolved cluster counts must be a pair of positive integers")
def bicluster_resolve_cluster_counts(n_clusters: int | tuple[int, int]) -> ClusterCounts:
    """Resolve row and column cluster counts from sklearn's biclustering parameter."""
    if isinstance(n_clusters, tuple):
        return int(n_clusters[0]), int(n_clusters[1])
    return int(n_clusters), int(n_clusters)


@register_atom(witness_bicluster_indicator_grid)
@icontract.require(
    lambda row_labels, column_labels, n_row_clusters, n_col_clusters: _indicator_grid_inputs_valid(
        row_labels, column_labels, n_row_clusters, n_col_clusters
    ),
    "row/column labels and cluster counts must be nonnegative and finite",
)
@icontract.ensure(
    lambda result, row_labels, column_labels, n_row_clusters, n_col_clusters: _indicator_grid_valid(
        result, row_labels, column_labels, n_row_clusters, n_col_clusters
    ),
    "indicator grid must produce boolean row and column membership matrices",
)
def bicluster_indicator_grid(
    row_labels: NDArray[np.int64],
    column_labels: NDArray[np.int64],
    n_row_clusters: int,
    n_col_clusters: int,
) -> BiclusterIndicatorGrid:
    """Build the repeated row and column indicator grids used by spectral biclustering."""
    row_values = np.asarray(row_labels, dtype=np.int64)
    col_values = np.asarray(column_labels, dtype=np.int64)
    rows = np.vstack(
        [
            row_values == label
            for label in range(n_row_clusters)
            for _ in range(n_col_clusters)
        ]
    )
    columns = np.vstack(
        [
            col_values == label
            for _ in range(n_row_clusters)
            for label in range(n_col_clusters)
        ]
    )
    return np.asarray(rows, dtype=np.bool_), np.asarray(columns, dtype=np.bool_)
