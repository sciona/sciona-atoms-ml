"""Partial-dependence column-slice shell atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_partial_dependence_slice_column_indices,
    witness_partial_dependence_slice_stop_exclusive,
    witness_partial_dependence_slice_uses_default_stop,
)


def _optional_nonnegative_int(value: object) -> bool:
    return value is None or (isinstance(value, int) and not isinstance(value, bool) and value >= 0)


def _positive_int(value: object) -> bool:
    return bool(isinstance(value, int) and not isinstance(value, bool) and value >= 1)


def _range_bounds_valid(start_location: int | None, stop_exclusive: int, n_columns: int) -> bool:
    start = 0 if start_location is None else int(start_location)
    return 0 <= start <= int(n_columns) and 0 <= int(stop_exclusive) <= int(n_columns) + 1


@register_atom(witness_partial_dependence_slice_uses_default_stop)
@icontract.require(lambda stop_location: _optional_nonnegative_int(stop_location), "stop_location must be None or a nonnegative integer")
@icontract.ensure(lambda result: isinstance(result, bool), "result must be boolean")
def partial_dependence_slice_uses_default_stop(
    stop_location: int | None,
) -> bool:
    """Decide whether sklearn uses `n_columns + 1` as the slice stop."""
    return stop_location is None


@register_atom(witness_partial_dependence_slice_stop_exclusive)
@icontract.require(lambda stop_location: _optional_nonnegative_int(stop_location), "stop_location must be None or a nonnegative integer")
@icontract.require(lambda n_columns: _positive_int(n_columns), "n_columns must be a positive integer")
@icontract.ensure(lambda result: isinstance(result, int) and result >= 1, "result must be a positive integer")
def partial_dependence_slice_stop_exclusive(
    stop_location: int | None,
    n_columns: int,
) -> int:
    """Resolve sklearn's exclusive slice stop after label-to-index conversion."""
    if stop_location is None:
        return int(n_columns) + 1
    return int(stop_location) + 1


@register_atom(witness_partial_dependence_slice_column_indices)
@icontract.require(lambda n_columns: _positive_int(n_columns), "n_columns must be a positive integer")
@icontract.require(lambda start_location: _optional_nonnegative_int(start_location), "start_location must be None or a nonnegative integer")
@icontract.require(lambda stop_exclusive: _positive_int(stop_exclusive), "stop_exclusive must be a positive integer")
@icontract.require(
    lambda start_location, stop_exclusive, n_columns: _range_bounds_valid(start_location, stop_exclusive, n_columns),
    "start_location, stop_exclusive, and n_columns must define a valid sklearn slice range",
)
@icontract.ensure(lambda result: isinstance(result, tuple) and all(isinstance(item, int) for item in result), "result must be a tuple of ints")
def partial_dependence_slice_column_indices(
    n_columns: int,
    start_location: int | None,
    stop_exclusive: int,
) -> tuple[int, ...]:
    """Build sklearn's integer column-index slice after label endpoints are resolved."""
    start = 0 if start_location is None else int(start_location)
    stop = int(stop_exclusive)
    return tuple(range(int(n_columns))[start:stop])
