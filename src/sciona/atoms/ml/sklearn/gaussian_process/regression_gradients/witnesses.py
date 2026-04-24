"""Ghost witnesses for Gaussian-process regression gradient helper atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def _check_matrix(values: AbstractArray, name: str) -> tuple[int, int]:
    if len(values.shape) != 2:
        raise ValueError(f"{name} must be 2D")
    rows, cols = int(values.shape[0]), int(values.shape[1])
    if rows < 1 or cols < 1:
        raise ValueError(f"{name} must be nonempty")
    return rows, cols


def _check_vector(values: AbstractArray, name: str) -> int:
    if len(values.shape) != 1:
        raise ValueError(f"{name} must be 1D")
    size = int(values.shape[0])
    if size < 1:
        raise ValueError(f"{name} must be nonempty")
    return size


def witness_gp_log_marginal_gradient_inner_term(
    L: AbstractArray,
    dual_coefficients: AbstractArray,
) -> AbstractArray:
    """Describe the GP log-marginal-likelihood inner tensor before kernel contraction."""
    n_samples, _ = _check_matrix(L, "L")
    if len(dual_coefficients.shape) == 1:
        if _check_vector(dual_coefficients, "dual_coefficients") != n_samples:
            raise ValueError("dual_coefficients must align with L")
        n_targets = 1
    else:
        rows, n_targets = _check_matrix(dual_coefficients, "dual_coefficients")
        if rows != n_samples:
            raise ValueError("dual_coefficients must align with L")
    return AbstractArray(shape=(n_samples, n_samples, n_targets), dtype="float64")


def witness_gp_log_marginal_gradient_dims(
    inner_term: AbstractArray,
    kernel_gradient: AbstractArray,
) -> AbstractArray:
    """Describe per-parameter per-output GP log-marginal-likelihood gradient terms."""
    if len(inner_term.shape) != 3:
        raise ValueError("inner_term must be 3D")
    n_samples = int(inner_term.shape[0])
    if int(inner_term.shape[1]) != n_samples:
        raise ValueError("inner_term must be square on its sample axes")
    if len(kernel_gradient.shape) != 3:
        raise ValueError("kernel_gradient must be 3D")
    if int(kernel_gradient.shape[0]) != n_samples or int(kernel_gradient.shape[1]) != n_samples:
        raise ValueError("kernel_gradient must align with the sample axes")
    return AbstractArray(shape=(int(kernel_gradient.shape[2]), int(inner_term.shape[2])), dtype="float64")


def witness_gp_log_marginal_gradient(
    gradient_dims: AbstractArray,
) -> AbstractArray:
    """Describe the final GP log-marginal-likelihood gradient vector."""
    if len(gradient_dims.shape) == 1:
        n_params = _check_vector(gradient_dims, "gradient_dims")
        return AbstractArray(shape=(n_params,), dtype="float64")
    n_params, _ = _check_matrix(gradient_dims, "gradient_dims")
    return AbstractArray(shape=(n_params,), dtype="float64")
