from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
from sciona.ghost.abstract import AbstractArray, AbstractScalar, AbstractSignal

def witness_extract_random_subspace_basis(A: AbstractArray, k: AbstractScalar | int, p: AbstractScalar | int, n_iter: AbstractScalar | int) -> AbstractArray:
    """Ghost witness for extract_random_subspace_basis."""
    _ = (A, k, p, n_iter)
    return AbstractArray(shape=A.shape, dtype=A.dtype)

def witness_factorize_subspace_projection(A: AbstractArray, Q: AbstractArray, k: AbstractScalar | int) -> AbstractArray:
    """Ghost witness for factorize_subspace_projection."""
    _ = (A, Q, k)
    return AbstractArray(shape=A.shape, dtype=A.dtype)

