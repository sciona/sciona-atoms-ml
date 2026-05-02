"""Ghost witnesses for Gaussian-process regression log-marginal-likelihood shell atoms."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.gaussian_process.kernels import Kernel


def witness_gp_log_marginal_likelihood_require_theta_for_gradient(
    theta_is_none: bool,
    eval_gradient: bool,
) -> None:
    """Describe the guard that rejects gradient evaluation when theta is missing."""
    del theta_is_none
    del eval_gradient
    return None


def witness_gp_log_marginal_likelihood_cached_result(
    log_marginal_likelihood_value: float,
) -> float:
    """Describe the cached log-marginal likelihood returned when theta is omitted."""
    del log_marginal_likelihood_value
    return 0.0


def witness_gp_log_marginal_likelihood_kernel(
    kernel: Kernel,
    theta: NDArray[np.float64],
    clone_kernel: bool,
) -> Kernel:
    """Describe the kernel object selected for log-marginal-likelihood evaluation."""
    del kernel
    del theta
    del clone_kernel
    raise NotImplementedError


def witness_gp_log_marginal_likelihood_cholesky_failure_result(
    theta: NDArray[np.float64],
    eval_gradient: bool,
) -> float | tuple[float, NDArray[np.float64]]:
    """Describe the fallback result returned on Cholesky failure."""
    del theta
    del eval_gradient
    return 0.0


def witness_gp_log_marginal_likelihood_train_targets(
    y_train: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Describe the target matrix used by log-marginal-likelihood evaluation."""
    del y_train
    return np.zeros((1, 1), dtype=np.float64)
