"""Ghost witnesses for permutation-importance preflight-shell atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_permutation_importance_use_dataframe_passthrough(
    has_iloc: bool,
) -> bool:
    """Describe the dataframe-passthrough branch predicate."""
    del has_iloc
    return False


def witness_permutation_importance_checked_array(
    X: AbstractArray,
) -> AbstractArray:
    """Describe the checked dense array used by permutation_importance."""
    if len(X.shape) != 2:
        raise ValueError("X must be 2D")
    return AbstractArray(shape=X.shape, dtype="float64")


def witness_permutation_importance_max_samples_guard_required(
    max_samples: int | float,
    n_samples: int,
) -> bool:
    """Describe the oversized-integer max_samples guard predicate."""
    del max_samples
    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    return False
