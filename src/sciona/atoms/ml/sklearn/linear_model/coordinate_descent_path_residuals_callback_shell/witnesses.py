"""Ghost witnesses for sklearn coordinate-descent path-residual callback-shell atoms."""

from __future__ import annotations


def witness_cd_path_residuals_check_array_accept_sparse(dtype: object) -> object:
    """Describe the fixed accept_sparse kwarg passed to check_array in _path_residuals."""
    return dtype


def witness_cd_path_residuals_check_array_dtype(dtype: object) -> object:
    """Describe the dtype kwarg passed to check_array in _path_residuals."""
    return dtype


def witness_cd_path_residuals_check_array_order(X_order: object) -> object:
    """Describe the order kwarg passed to check_array in _path_residuals."""
    return X_order


def witness_cd_path_residuals_path_result_alphas(path_result: object) -> object:
    """Describe the alphas component unpacked from path(...) in _path_residuals."""
    return path_result


def witness_cd_path_residuals_path_result_coefs(path_result: object) -> object:
    """Describe the coefficient component unpacked from path(...) in _path_residuals."""
    return path_result
