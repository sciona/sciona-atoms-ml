"""Finite matrix checks for biclustering SVD helpers."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_bicluster_svd_checked_u,
    witness_bicluster_svd_checked_vt,
)

def _nonempty_matrix(values: object) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 2 and array.shape[0] >= 1 and array.shape[1] >= 1)

def _finite_matrix(values: object) -> bool:
    if not _nonempty_matrix(values):
        return False
    array = np.asarray(values, dtype=np.float64)
    return bool(np.all(np.isfinite(array)))

def _same_shape_and_values(result: object, expected: object) -> bool:
    lhs = np.asarray(result, dtype=np.float64)
    rhs = np.asarray(expected, dtype=np.float64)
    return bool(lhs.shape == rhs.shape and np.array_equal(lhs, rhs))

@register_atom(witness_bicluster_svd_checked_u)
@icontract.require(lambda u: _nonempty_matrix(u), "u must be a nonempty 2D float-like matrix")
@icontract.ensure(
    lambda result, u: _finite_matrix(result) and _same_shape_and_values(result, np.asarray(u, dtype=np.float64)),
    "result must preserve the finite u matrix",
)
def bicluster_svd_checked_u(
    u: NDArray[np.float64],
) -> NDArray[np.float64]:
    from sklearn.utils.validation import assert_all_finite
    """Validate and expose the finite left singular-vector matrix in BaseSpectral._svd."""
    values = np.asarray(u, dtype=np.float64)
    assert_all_finite(values)
    return values

@register_atom(witness_bicluster_svd_checked_vt)
@icontract.require(lambda vt: _nonempty_matrix(vt), "vt must be a nonempty 2D float-like matrix")
@icontract.ensure(
    lambda result, vt: _finite_matrix(result) and _same_shape_and_values(result, np.asarray(vt, dtype=np.float64)),
    "result must preserve the finite vt matrix",
)
def bicluster_svd_checked_vt(
    vt: NDArray[np.float64],
) -> NDArray[np.float64]:
    from sklearn.utils.validation import assert_all_finite
    """Validate and expose the finite right singular-vector matrix in BaseSpectral._svd."""
    values = np.asarray(vt, dtype=np.float64)
    assert_all_finite(values)
    return values
