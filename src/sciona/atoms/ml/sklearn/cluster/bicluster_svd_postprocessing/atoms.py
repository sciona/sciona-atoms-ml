"""Helpers for deterministic biclustering SVD output shaping adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_bicluster_svd_left_vectors,
    witness_bicluster_svd_right_vectors,
)


def _finite_matrix(values: object) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(
        array.ndim == 2
        and array.shape[0] >= 1
        and array.shape[1] >= 1
        and np.all(np.isfinite(array))
    )


def _valid_n_discard_for_columns(values: object, n_discard: object) -> bool:
    if not _finite_matrix(values):
        return False
    if not isinstance(n_discard, int) or isinstance(n_discard, bool):
        return False
    array = np.asarray(values, dtype=np.float64)
    return bool(0 <= int(n_discard) < array.shape[1])


def _valid_n_discard_for_rows(values: object, n_discard: object) -> bool:
    if not _finite_matrix(values):
        return False
    if not isinstance(n_discard, int) or isinstance(n_discard, bool):
        return False
    array = np.asarray(values, dtype=np.float64)
    return bool(0 <= int(n_discard) < array.shape[0])


def _same_shape_and_values(result: object, expected: object) -> bool:
    lhs = np.asarray(result, dtype=np.float64)
    rhs = np.asarray(expected, dtype=np.float64)
    return bool(lhs.shape == rhs.shape and np.array_equal(lhs, rhs))


@register_atom(witness_bicluster_svd_left_vectors)
@icontract.require(
    lambda u, n_discard: _valid_n_discard_for_columns(u, n_discard),
    "u must be a finite nonempty matrix and n_discard must leave at least one column",
)
@icontract.ensure(
    lambda result, u, n_discard: _same_shape_and_values(result, np.asarray(u, dtype=np.float64)[:, int(n_discard):]),
    "result must match the kept left singular-vector columns",
)
def bicluster_svd_left_vectors(
    u: NDArray[np.float64],
    n_discard: int,
) -> NDArray[np.float64]:
    """Slice the kept left singular vectors returned by BaseSpectral._svd."""
    values = np.asarray(u, dtype=np.float64)
    return np.asarray(values[:, int(n_discard):], dtype=np.float64)


@register_atom(witness_bicluster_svd_right_vectors)
@icontract.require(
    lambda vt, n_discard: _valid_n_discard_for_rows(vt, n_discard),
    "vt must be a finite nonempty matrix and n_discard must leave at least one row",
)
@icontract.ensure(
    lambda result, vt, n_discard: _same_shape_and_values(result, np.asarray(vt, dtype=np.float64)[int(n_discard):].T),
    "result must match the transposed kept right singular vectors",
)
def bicluster_svd_right_vectors(
    vt: NDArray[np.float64],
    n_discard: int,
) -> NDArray[np.float64]:
    """Slice and transpose the kept right singular vectors returned by BaseSpectral._svd."""
    values = np.asarray(vt, dtype=np.float64)
    return np.asarray(values[int(n_discard):].T, dtype=np.float64)
