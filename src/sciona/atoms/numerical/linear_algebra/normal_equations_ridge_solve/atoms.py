from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
import numpy as np
from numpy.typing import NDArray
import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_compute_gram_matrix,
    witness_apply_tikhonov_shift_and_solve,
)

@register_atom(witness_compute_gram_matrix, name="compute_gram_matrix")
@icontract.require(lambda A, b: A.ndim == 2, "Precondition failed: A.ndim == 2")
@icontract.require(lambda A, b: A.shape[0] == b.shape[0], "Precondition failed: A.shape[0] == b.shape[0]")
@icontract.ensure(lambda result, A, b: Gram.shape[0] == Gram.shape[1], "Postcondition failed: Gram.shape[0] == Gram.shape[1]")
def compute_gram_matrix(A: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
    """Assemble Gram matrix and projected RHS.

    Args:
        A: NDArray[np.float64]
        b: NDArray[np.float64]

    Returns:
        Gram: NDArray[np.float64]
    """
    import numpy
    return numpy.dot(A=A, b=b) # type: ignore

@register_atom(witness_apply_tikhonov_shift_and_solve, name="apply_tikhonov_shift_and_solve")
@icontract.require(lambda Gram, Ab, alpha: alpha >= 0.0, "Precondition failed: alpha >= 0.0")
@icontract.ensure(lambda result, Gram, Ab, alpha: x.shape == Ab.shape, "Postcondition failed: x.shape == Ab.shape")
def apply_tikhonov_shift_and_solve(Gram: NDArray[np.float64], Ab: NDArray[np.float64], alpha: float) -> NDArray[np.float64]:
    """Apply diagonal regularization shift and solve via Cholesky.

    Args:
        Gram: NDArray[np.float64]
        Ab: NDArray[np.float64]
        alpha: alpha >= 0

    Returns:
        x: NDArray[np.float64]
    """
    import scipy.linalg
    return scipy.linalg.cho_factor(Gram=Gram, Ab=Ab, alpha=alpha) # type: ignore

