from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
import numpy as np
from numpy.typing import NDArray
import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_compute_laplacian_matrix,
    witness_solve_smallest_eigen,
)

@register_atom(witness_compute_laplacian_matrix, name="compute_laplacian_matrix")
@icontract.require(lambda adj_matrix, normalized: adj_matrix.shape[0] == adj_matrix.shape[1], "Precondition failed: adj_matrix.shape[0] == adj_matrix.shape[1]")
@icontract.ensure(lambda result, adj_matrix, normalized: result is not None, "Postcondition failed: result is not None")
def compute_laplacian_matrix(adj_matrix: scipy.sparse.csr_array, normalized: bool) -> scipy.sparse.csr_array:
    """Build graph Laplacian L = D - A.

    Args:
        adj_matrix: scipy.sparse.csr_array
        normalized: bool

    Returns:
        laplacian: scipy.sparse.csr_array
    """
    import scipy.sparse.csgraph
    return scipy.sparse.csgraph.laplacian(adj_matrix=adj_matrix, normalized=normalized) # type: ignore

@register_atom(witness_solve_smallest_eigen, name="solve_smallest_eigen")
@icontract.require(lambda laplacian, k: k < laplacian.shape[0], "Precondition failed: k < laplacian.shape[0]")
@icontract.ensure(lambda result, laplacian, k: eigenvalues.shape[0] == k, "Postcondition failed: eigenvalues.shape[0] == k")
@icontract.ensure(lambda result, laplacian, k: eigenvectors.shape == (laplacian.shape[0], k), "Postcondition failed: eigenvectors.shape == (laplacian.shape[0], k)")
def solve_smallest_eigen(laplacian: scipy.sparse.csr_array, k: int) -> NDArray[np.float64]:
    """Extract d+1 smallest eigenvalues/eigenvectors using sparse solvers.

    Args:
        laplacian: scipy.sparse.csr_array
        k: int

    Returns:
        eigenvalues: NDArray[np.float64]
    """
    import scipy.sparse.linalg
    return scipy.sparse.linalg.eigsh(laplacian=laplacian, k=k) # type: ignore

