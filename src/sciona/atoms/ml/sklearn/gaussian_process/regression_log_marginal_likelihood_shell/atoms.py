"""Gaussian-process regression log-marginal-likelihood shell atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_gp_log_marginal_likelihood_cached_result,
    witness_gp_log_marginal_likelihood_cholesky_failure_result,
    witness_gp_log_marginal_likelihood_kernel,
    witness_gp_log_marginal_likelihood_require_theta_for_gradient,
    witness_gp_log_marginal_likelihood_train_targets,
)

def _bool(value: object) -> bool:
    return isinstance(value, bool)

def _finite_scalar(value: object) -> bool:
    return bool(np.isscalar(value) and not isinstance(value, bool) and np.isfinite(float(value)))

def _finite_vector(values: object) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 1 and array.shape[0] >= 1 and np.all(np.isfinite(array)))

def _finite_vector_or_matrix(values: object) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(
        array.ndim in {1, 2}
        and array.shape[0] >= 1
        and (array.ndim == 1 or array.shape[1] >= 1)
        and np.all(np.isfinite(array))
    )

def _kernel(value: object) -> bool:
    from sklearn.gaussian_process.kernels import Kernel
    return isinstance(value, Kernel)

def _failure_result_valid(result: object, theta: NDArray[np.float64], eval_gradient: bool) -> bool:
    if eval_gradient:
        if not isinstance(result, tuple) or len(result) != 2:
            return False
        log_value, gradient = result
        return bool(
            float(log_value) == -np.inf
            and np.array_equal(np.asarray(gradient, dtype=np.float64), np.zeros_like(np.asarray(theta, dtype=np.float64)))
        )
    return bool(np.isscalar(result) and float(result) == -np.inf)

def _train_targets_valid(result: object, y_train: object) -> bool:
    try:
        values = np.asarray(result, dtype=np.float64)
        source = np.asarray(y_train, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    if source.ndim == 1:
        return bool(values.shape == (source.shape[0], 1) and np.array_equal(values[:, 0], source))
    return bool(values.shape == source.shape and np.array_equal(values, source))

@register_atom(witness_gp_log_marginal_likelihood_require_theta_for_gradient)
@icontract.require(lambda theta_is_none: _bool(theta_is_none), "theta_is_none must be boolean")
@icontract.require(lambda eval_gradient: _bool(eval_gradient), "eval_gradient must be boolean")
@icontract.ensure(lambda result: result is None, "guard returns None when it does not raise")
def gp_log_marginal_likelihood_require_theta_for_gradient(
    theta_is_none: bool,
    eval_gradient: bool,
) -> None:
    """Reject gradient evaluation when theta is omitted."""
    if theta_is_none and eval_gradient:
        raise ValueError("Gradient can only be evaluated for theta!=None")

@register_atom(witness_gp_log_marginal_likelihood_cached_result)
@icontract.require(lambda log_marginal_likelihood_value: _finite_scalar(log_marginal_likelihood_value), "cached log-marginal-likelihood value must be finite")
@icontract.ensure(lambda result, log_marginal_likelihood_value: _finite_scalar(result) and float(result) == float(log_marginal_likelihood_value), "result must preserve the cached log-marginal-likelihood value")
def gp_log_marginal_likelihood_cached_result(
    log_marginal_likelihood_value: float,
) -> float:
    """Return the cached log-marginal likelihood for theta=None without gradient."""
    return float(log_marginal_likelihood_value)

@register_atom(witness_gp_log_marginal_likelihood_kernel)
@icontract.require(lambda kernel: _kernel(kernel), "kernel must be a sklearn kernel instance")
@icontract.require(lambda theta: _finite_vector(theta), "theta must be a finite nonempty vector")
@icontract.require(lambda clone_kernel: _bool(clone_kernel), "clone_kernel must be boolean")
@icontract.ensure(lambda result: _kernel(result), "result must be a sklearn kernel instance")
@icontract.ensure(
    lambda result, kernel, theta, clone_kernel: (
        result is not kernel and np.array_equal(np.asarray(result.theta, dtype=np.float64), np.asarray(theta, dtype=np.float64))
    ) if clone_kernel else (
        result is kernel and np.array_equal(np.asarray(result.theta, dtype=np.float64), np.asarray(theta, dtype=np.float64))
    ),
    "result must be a clone-with-theta when clone_kernel is true, or the supplied kernel with theta assigned in place otherwise",
)
def gp_log_marginal_likelihood_kernel(
    kernel: Kernel,
    theta: NDArray[np.float64],
    clone_kernel: bool,
) -> Kernel:
    from sklearn.gaussian_process.kernels import Kernel
    """Resolve the kernel object used for log-marginal-likelihood evaluation."""
    theta_values = np.asarray(theta, dtype=np.float64)
    if clone_kernel:
        return kernel.clone_with_theta(theta_values)
    kernel.theta = theta_values
    return kernel

@register_atom(witness_gp_log_marginal_likelihood_cholesky_failure_result)
@icontract.require(lambda theta: _finite_vector(theta), "theta must be a finite nonempty vector")
@icontract.require(lambda eval_gradient: _bool(eval_gradient), "eval_gradient must be boolean")
@icontract.ensure(
    lambda result, theta, eval_gradient: _failure_result_valid(result, theta, eval_gradient),
    "failure result must match sklearn's -inf or (-inf, zeros_like(theta)) fallback",
)
def gp_log_marginal_likelihood_cholesky_failure_result(
    theta: NDArray[np.float64],
    eval_gradient: bool,
) -> float | tuple[float, NDArray[np.float64]]:
    """Return sklearn's Cholesky-failure fallback result for log-marginal likelihood."""
    if eval_gradient:
        return -np.inf, np.zeros_like(np.asarray(theta, dtype=np.float64))
    return float(-np.inf)

@register_atom(witness_gp_log_marginal_likelihood_train_targets)
@icontract.require(lambda y_train: _finite_vector_or_matrix(y_train), "y_train must be a finite nonempty vector or matrix")
@icontract.ensure(lambda result, y_train: _train_targets_valid(result, y_train), "result must preserve 2D targets and expand 1D targets into a column matrix")
def gp_log_marginal_likelihood_train_targets(
    y_train: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Expand one-dimensional training targets into the column-matrix form used by log-marginal-likelihood evaluation."""
    values = np.asarray(y_train, dtype=np.float64)
    if values.ndim == 1:
        return values[:, np.newaxis]
    return values
