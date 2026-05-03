"""Ghost witnesses for binary Gaussian-process classification log-marginal-likelihood shell atoms."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.gaussian_process.kernels import Kernel


def witness_gpc_binary_log_marginal_likelihood_require_theta_for_gradient(
    theta_is_none: bool,
    eval_gradient: bool,
) -> None:
    """Describe the theta-required-for-gradient guard."""
    del theta_is_none
    del eval_gradient
    return None


def witness_gpc_binary_log_marginal_likelihood_cached_result(
    log_marginal_likelihood_value: float,
) -> float:
    """Describe the cached log-marginal-likelihood return value."""
    return float(log_marginal_likelihood_value)


def witness_gpc_binary_log_marginal_likelihood_kernel(
    kernel: Kernel,
    theta: NDArray[np.float64],
    clone_kernel: bool,
) -> Kernel:
    """Describe the kernel object used for binary GPC log-marginal-likelihood evaluation."""
    del theta
    del clone_kernel
    return kernel


def witness_gpc_binary_log_marginal_likelihood_use_gradient_branch(
    eval_gradient: bool,
) -> bool:
    """Describe whether binary GPC log-marginal-likelihood uses the gradient branch."""
    return bool(eval_gradient)


def witness_gpc_binary_log_marginal_likelihood_result(
    Z: float,
    eval_gradient: bool,
    *,
    gradient: NDArray[np.float64] | None = None,
) -> float | tuple[float, NDArray[np.float64]]:
    """Describe the scalar-or-tuple return packaging."""
    if eval_gradient:
        if gradient is None:
            return float(Z), np.zeros(1, dtype=np.float64)
        return float(Z), np.asarray(gradient, dtype=np.float64)
    return float(Z)
