"""Ghost witnesses for partial-dependence input-shell atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_partial_dependence_use_object_check_array(
    has_array: bool,
    is_sparse: bool,
) -> bool:
    """Describe the non-array-like object check_array branch predicate."""
    del has_array, is_sparse
    return False


def witness_partial_dependence_checked_object_array(
    X: AbstractArray,
) -> AbstractArray:
    """Describe the checked object array used by partial_dependence."""
    if len(X.shape) != 2:
        raise ValueError("X must be 2D")
    return AbstractArray(shape=X.shape, dtype="object")
