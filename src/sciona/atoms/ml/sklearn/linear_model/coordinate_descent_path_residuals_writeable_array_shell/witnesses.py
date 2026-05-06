"""Ghost witnesses for sklearn coordinate-descent path-residual writeable-array atoms."""

from __future__ import annotations


def witness_cd_path_residuals_dense_writeable_guard(X_is_sparse: object) -> object:
    """Describe the dense-only writeable-array loop gate in _path_residuals."""
    return X_is_sparse


def witness_cd_path_residuals_array_needs_writeable_fix(
    array_base_matches_input: object, array_writeable: object
) -> object:
    """Describe the per-array memmap writeability guard in _path_residuals."""
    return array_base_matches_input, array_writeable


def witness_cd_path_residuals_writable_array(array: object, array_input: object) -> object:
    """Describe the array.setflags(write=True) normalization shell in _path_residuals."""
    return array, array_input
