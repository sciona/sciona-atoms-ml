"""Ghost witnesses for spectral biclustering normalization-dispatch atoms."""

from __future__ import annotations

import scipy.sparse as sp
from numpy.typing import NDArray


def witness_bicluster_dense_normalized_data(
    X: NDArray[float],
    method: str,
) -> NDArray[float]:
    """Describe the dense normalized matrix selected by spectral biclustering."""
    del method
    return X


def witness_bicluster_sparse_normalized_data(
    X: sp.spmatrix,
    method: str,
) -> sp.spmatrix:
    """Describe the sparse normalized matrix selected by spectral biclustering."""
    del method
    return X
