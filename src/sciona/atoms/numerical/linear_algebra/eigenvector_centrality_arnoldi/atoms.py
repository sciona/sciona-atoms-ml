from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
import numpy as np
from numpy.typing import NDArray
import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_solve_dominant_eigenvector,
    witness_perron_frobenius_correct,
)

@register_atom(witness_solve_dominant_eigenvector, name="solve_dominant_eigenvector")
@icontract.require(lambda adj_matrix, max_iter: max_iter > 0, "Precondition failed: max_iter > 0")
@icontract.ensure(lambda result, adj_matrix, max_iter: result is not None, "Postcondition failed: result is not None")
def solve_dominant_eigenvector(adj_matrix: scipy.sparse.csr_array, max_iter: int) -> Any:
    """Call ARPACK to extract the largest eigenvalue/vector.

    Args:
        adj_matrix: scipy.sparse.csr_array
        max_iter: int

    Returns:
        eigenvalue: complex
    """
    import scipy.sparse.linalg
    return scipy.sparse.linalg.eigs(adj_matrix=adj_matrix, max_iter=max_iter) # type: ignore

@register_atom(witness_perron_frobenius_correct, name="perron_frobenius_correct")
@icontract.require(lambda eigenvector: eigenvector is not None, "Precondition failed: eigenvector is not None")
@icontract.ensure(lambda result, eigenvector: result is not None, "Postcondition failed: result is not None")
def perron_frobenius_correct(eigenvector: NDArray[np.complex128]) -> NDArray[np.float64]:
    """Cast complex values, verify positiveness, and normalize vector.

    Args:
        eigenvector: NDArray[np.complex128]

    Returns:
        scores: NDArray[np.float64]
    """
    import numpy
    return numpy.real(eigenvector=eigenvector) # type: ignore

