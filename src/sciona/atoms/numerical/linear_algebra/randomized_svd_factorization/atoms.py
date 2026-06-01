from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
import numpy as np
from numpy.typing import NDArray
import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_extract_random_subspace_basis,
    witness_factorize_subspace_projection,
)

@register_atom(witness_extract_random_subspace_basis, name="extract_random_subspace_basis")
@icontract.require(lambda A, k, p, n_iter: k > 0, "Precondition failed: k > 0")
@icontract.require(lambda A, k, p, n_iter: p >= 0, "Precondition failed: p >= 0")
@icontract.ensure(lambda result, A, k, p, n_iter: Q.shape[0] == A.shape[0], "Postcondition failed: Q.shape[0] == A.shape[0]")
def extract_random_subspace_basis(A: NDArray[np.float64], k: int, p: int, n_iter: int) -> NDArray[np.float64]:
    """Form random projection basis Q for A.

    Args:
        A: NDArray[np.float64]
        k: int
        p: int
        n_iter: int

    Returns:
        Q: NDArray[np.float64]
    """
    import sklearn.utils.extmath
    return sklearn.utils.extmath.randomized_svd(A=A, k=k, p=p, n_iter=n_iter) # type: ignore

@register_atom(witness_factorize_subspace_projection, name="factorize_subspace_projection")
@icontract.require(lambda A, Q, k: Q.shape[0] == A.shape[0], "Precondition failed: Q.shape[0] == A.shape[0]")
@icontract.ensure(lambda result, A, Q, k: s_k.shape[0] == k, "Postcondition failed: s_k.shape[0] == k")
def factorize_subspace_projection(A: NDArray[np.float64], Q: NDArray[np.float64], k: int) -> NDArray[np.float64]:
    """Project A onto Q and perform core dense SVD.

    Args:
        A: NDArray[np.float64]
        Q: NDArray[np.float64]
        k: int

    Returns:
        U_k: NDArray[np.float64]
    """
    import scipy.linalg
    return scipy.linalg.svd(A=A, Q=Q, k=k) # type: ignore

