"""Ghost witnesses for partial-dependence column-slice shell atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_partial_dependence_slice_uses_default_stop(
    stop_location: int | None,
) -> AbstractArray:
    """Describe the default-stop predicate for label slices."""
    del stop_location
    return AbstractArray(shape=(), dtype="bool")


def witness_partial_dependence_slice_stop_exclusive(
    stop_location: int | None,
    n_columns: int,
) -> AbstractArray:
    """Describe sklearn's exclusive slice stop after label resolution."""
    del stop_location
    return AbstractArray(shape=(), dtype="int64", min_val=1.0, max_val=float(n_columns + 1))


def witness_partial_dependence_slice_column_indices(
    n_columns: int,
    start_location: int | None,
    stop_exclusive: int,
) -> AbstractArray:
    """Describe the resulting integer slice indices."""
    del start_location
    if int(n_columns) < 1:
        raise ValueError("n_columns must be positive")
    if int(stop_exclusive) < 1:
        raise ValueError("stop_exclusive must be positive")
    return AbstractArray(shape=(int(n_columns),), dtype="int64")
