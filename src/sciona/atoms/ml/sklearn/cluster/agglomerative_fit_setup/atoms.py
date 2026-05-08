"""Agglomerative fit setup atoms adapted from scikit-learn."""

from __future__ import annotations

from collections.abc import Callable

import icontract
import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_agglomerative_fit_prepare_connectivity,
    witness_agglomerative_fit_select_tree_builder,
)

MatrixLike = NDArray[np.float64] | sp.spmatrix | list[list[float]]
ConnectivityInput = MatrixLike | Callable[[MatrixLike], MatrixLike] | None

def _is_2d_matrix(values: MatrixLike) -> bool:
    if sp.issparse(values):
        return bool(values.ndim == 2 and values.shape[0] >= 1 and values.shape[1] >= 1)
    try:
        array = np.asarray(values)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 2 and array.shape[0] >= 1 and array.shape[1] >= 1)

def _sample_count(values: MatrixLike) -> int:
    return int(values.shape[0]) if sp.issparse(values) else int(np.asarray(values).shape[0])

def _linkage_valid(linkage: str) -> bool:
    from sklearn.cluster import _agglomerative as sklearn_agglomerative
    return bool(isinstance(linkage, str) and linkage in sklearn_agglomerative._TREE_BUILDERS)

def _prepared_connectivity_valid(result: object, X: MatrixLike) -> bool:
    if result is None:
        return True
    if sp.issparse(result):
        return bool(
            result.ndim == 2
            and result.shape[0] == result.shape[1] == _sample_count(X)
            and result.format in {"csr", "coo", "lil"}
            and np.all(np.isfinite(result.data))
        )
    try:
        array = np.asarray(result, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(
        array.ndim == 2
        and array.shape[0] == array.shape[1] == _sample_count(X)
        and np.all(np.isfinite(array))
    )

@register_atom(witness_agglomerative_fit_select_tree_builder)
@icontract.require(lambda linkage: _linkage_valid(linkage), "linkage must map to a known sklearn agglomerative tree builder")
@icontract.ensure(lambda result: callable(result), "tree builder must be callable")
def agglomerative_fit_select_tree_builder(
    linkage: str,
) -> Callable[..., object]:
    from sklearn.cluster import _agglomerative as sklearn_agglomerative
    """Select sklearn's agglomerative tree builder for the given linkage."""
    return sklearn_agglomerative._TREE_BUILDERS[linkage]

@register_atom(witness_agglomerative_fit_prepare_connectivity)
@icontract.require(lambda X: _is_2d_matrix(X), "X must be a nonempty 2D feature or distance matrix")
@icontract.ensure(lambda result, X: _prepared_connectivity_valid(result, X), "prepared connectivity must be None or a finite square matrix matching X")
def agglomerative_fit_prepare_connectivity(
    X: MatrixLike,
    connectivity: ConnectivityInput,
) -> MatrixLike | None:
    from sklearn.utils.validation import check_array
    """Resolve optional agglomerative connectivity input through callable execution and check_array normalization."""
    if connectivity is None:
        return None
    resolved = connectivity(X) if callable(connectivity) else connectivity
    return check_array(resolved, accept_sparse=["csr", "coo", "lil"])
