"""Binary Gaussian-process classification log-marginal-likelihood shell atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_gpc_binary_log_marginal_likelihood_cached_result,
    witness_gpc_binary_log_marginal_likelihood_kernel,
    witness_gpc_binary_log_marginal_likelihood_require_theta_for_gradient,
    witness_gpc_binary_log_marginal_likelihood_result,
    witness_gpc_binary_log_marginal_likelihood_use_gradient_branch,
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

def _kernel(value: object) -> bool:
    from sklearn.gaussian_process.kernels import Kernel
    return isinstance(value, Kernel)

def _kernel_selection_valid(
    result: object,
    kernel: Kernel,
    theta: NDArray[np.float64],
    clone_kernel: bool,
) -> bool:
    from sklearn.gaussian_process.kernels import Kernel
    if not _kernel(result):
        return False
    result_kernel = result
    expected_theta = np.asarray(theta, dtype=np.float64)
    observed_theta = np.asarray(result_kernel.theta, dtype=np.float64)
    if clone_kernel:
        return bool(result_kernel is not kernel and np.allclose(observed_theta, expected_theta))
    return bool(result_kernel is kernel and np.allclose(observed_theta, expected_theta))

def _result_valid(
    result: object,
    Z: float,
    eval_gradient: bool,
    gradient: NDArray[np.float64] | None,
) -> bool:
    if eval_gradient:
        if gradient is None or not isinstance(result, tuple) or len(result) != 2:
            return False
        log_value, observed_gradient = result
        return bool(
            float(log_value) == float(Z)
            and np.array_equal(np.asarray(observed_gradient, dtype=np.float64), np.asarray(gradient, dtype=np.float64))
        )
    return bool(np.isscalar(result) and float(result) == float(Z))

@register_atom(witness_gpc_binary_log_marginal_likelihood_require_theta_for_gradient)
@icontract.require(lambda theta_is_none: _bool(theta_is_none), "theta_is_none must be boolean")
@icontract.require(lambda eval_gradient: _bool(eval_gradient), "eval_gradient must be boolean")
@icontract.ensure(lambda result: result is None, "guard returns None when it does not raise")
def gpc_binary_log_marginal_likelihood_require_theta_for_gradient(
    theta_is_none: bool,
    eval_gradient: bool,
) -> None:
    """Reject gradient evaluation when theta is omitted."""
    if theta_is_none and eval_gradient:
        raise ValueError("Gradient can only be evaluated for theta!=None")

@register_atom(witness_gpc_binary_log_marginal_likelihood_cached_result)
@icontract.require(lambda log_marginal_likelihood_value: _finite_scalar(log_marginal_likelihood_value), "cached log-marginal-likelihood value must be finite")
@icontract.ensure(lambda result, log_marginal_likelihood_value: _finite_scalar(result) and float(result) == float(log_marginal_likelihood_value), "result must preserve the cached log-marginal-likelihood value")
def gpc_binary_log_marginal_likelihood_cached_result(
    log_marginal_likelihood_value: float,
) -> float:
    """Return the cached log-marginal likelihood for theta=None without gradient."""
    return float(log_marginal_likelihood_value)

@register_atom(witness_gpc_binary_log_marginal_likelihood_kernel)
@icontract.require(lambda kernel: _kernel(kernel), "kernel must be a sklearn kernel instance")
@icontract.require(lambda theta: _finite_vector(theta), "theta must be a finite nonempty vector")
@icontract.require(lambda clone_kernel: _bool(clone_kernel), "clone_kernel must be boolean")
@icontract.ensure(lambda result: _kernel(result), "result must be a sklearn kernel instance")
@icontract.ensure(
    lambda result, kernel, theta, clone_kernel: _kernel_selection_valid(result, kernel, theta, clone_kernel),
    "result must be a clone-with-theta when clone_kernel is true, or the supplied kernel with theta assigned in place otherwise",
)
def gpc_binary_log_marginal_likelihood_kernel(
    kernel: Kernel,
    theta: NDArray[np.float64],
    clone_kernel: bool,
) -> Kernel:
    from sklearn.gaussian_process.kernels import Kernel
    """Resolve the kernel object used for binary GPC log-marginal-likelihood evaluation."""
    theta_values = np.asarray(theta, dtype=np.float64)
    if clone_kernel:
        return kernel.clone_with_theta(theta_values)
    kernel.theta = theta_values
    return kernel

@register_atom(witness_gpc_binary_log_marginal_likelihood_use_gradient_branch)
@icontract.require(lambda eval_gradient: _bool(eval_gradient), "eval_gradient must be boolean")
@icontract.ensure(lambda result: _bool(result), "result must be boolean")
def gpc_binary_log_marginal_likelihood_use_gradient_branch(
    eval_gradient: bool,
) -> bool:
    """Decide whether binary GPC log-marginal-likelihood uses the kernel eval_gradient branch."""
    return bool(eval_gradient)

@register_atom(witness_gpc_binary_log_marginal_likelihood_result)
@icontract.require(lambda Z: _finite_scalar(Z), "Z must be a finite scalar")
@icontract.require(lambda eval_gradient: _bool(eval_gradient), "eval_gradient must be boolean")
@icontract.require(
    lambda gradient=None, eval_gradient=False: (gradient is None and not eval_gradient) or (gradient is not None and _finite_vector(gradient)),
    "gradient must be a finite vector exactly when eval_gradient is true",
)
@icontract.ensure(
    lambda result, Z, eval_gradient, gradient=None: _result_valid(result, Z, eval_gradient, gradient),
    "result must match sklearn's scalar-or-tuple return packaging",
)
def gpc_binary_log_marginal_likelihood_result(
    Z: float,
    eval_gradient: bool,
    *,
    gradient: NDArray[np.float64] | None = None,
) -> float | tuple[float, NDArray[np.float64]]:
    """Package binary GPC log-marginal-likelihood outputs in sklearn's return shape."""
    if eval_gradient:
        assert gradient is not None
        return float(Z), np.asarray(gradient, dtype=np.float64)
    return float(Z)
