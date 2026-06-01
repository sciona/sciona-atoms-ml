from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
from sciona.ghost.abstract import AbstractArray, AbstractScalar, AbstractSignal

def witness_compute_gram_matrix(A: AbstractArray, b: AbstractArray) -> AbstractArray:
    """Ghost witness for compute_gram_matrix."""
    _ = (A, b)
    return AbstractArray(shape=A.shape, dtype=A.dtype)

def witness_apply_tikhonov_shift_and_solve(Gram: AbstractArray, Ab: AbstractArray, alpha: AbstractScalar | float) -> AbstractArray:
    """Ghost witness for apply_tikhonov_shift_and_solve."""
    _ = (Gram, Ab, alpha)
    return AbstractArray(shape=Gram.shape, dtype=Gram.dtype)

