"""Ghost witnesses for Gaussian-process kernel hyperparameter helpers."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def _check_parallel_lengths(*items: tuple[object, ...]) -> int:
    lengths = {len(item) for item in items}
    if len(lengths) != 1:
        raise ValueError("input tuples must have matching lengths")
    return lengths.pop()


def witness_gp_kernel_theta(
    value_blocks: tuple[AbstractArray, ...],
    fixed: tuple[bool, ...],
) -> AbstractArray:
    """Describe the flattened log-theta vector for non-fixed kernel hyperparameters."""
    _check_parallel_lengths(value_blocks, fixed)
    n_dims = 0
    for block, is_fixed in zip(value_blocks, fixed):
        if len(block.shape) != 1:
            raise ValueError("value blocks must be one-dimensional")
        if not is_fixed:
            n_dims += int(block.shape[0])
    return AbstractArray(shape=(n_dims,), dtype="float64")


def witness_gp_kernel_values_from_theta(
    theta: AbstractArray,
    value_blocks: tuple[AbstractArray, ...],
    fixed: tuple[bool, ...],
) -> tuple[AbstractArray, ...]:
    """Describe kernel hyperparameter value blocks reconstructed from theta."""
    _check_parallel_lengths(value_blocks, fixed)
    if len(theta.shape) != 1:
        raise ValueError("theta must be one-dimensional")
    expected_dims = sum(int(block.shape[0]) for block, is_fixed in zip(value_blocks, fixed) if not is_fixed)
    if int(theta.shape[0]) != expected_dims:
        raise ValueError("theta length must match the number of non-fixed hyperparameter elements")
    return tuple(AbstractArray(shape=block.shape, dtype="float64") for block in value_blocks)


def witness_gp_kernel_bounds(
    bounds_blocks: tuple[AbstractArray, ...],
    fixed: tuple[bool, ...],
) -> AbstractArray:
    """Describe the flattened log-bounds matrix for non-fixed kernel hyperparameters."""
    _check_parallel_lengths(bounds_blocks, fixed)
    n_dims = 0
    for block, is_fixed in zip(bounds_blocks, fixed):
        if len(block.shape) != 2 or int(block.shape[1]) != 2:
            raise ValueError("bounds blocks must be two-dimensional with width 2")
        if not is_fixed:
            n_dims += int(block.shape[0])
    if n_dims == 0:
        return AbstractArray(shape=(0,), dtype="float64")
    return AbstractArray(shape=(n_dims, 2), dtype="float64")


def witness_gp_kernel_bound_warning_records(
    theta: AbstractArray,
    bounds_blocks: tuple[AbstractArray, ...],
    fixed: tuple[bool, ...],
    names: tuple[str, ...],
) -> tuple[tuple[str, int, str, float], ...]:
    """Describe convergence-warning records for hyperparameters close to their bounds."""
    _check_parallel_lengths(bounds_blocks, fixed, names)
    if len(theta.shape) != 1:
        raise ValueError("theta must be one-dimensional")
    return ()
