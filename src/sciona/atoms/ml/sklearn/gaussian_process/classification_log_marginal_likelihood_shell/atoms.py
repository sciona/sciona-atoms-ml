"""Gaussian-process classification log-marginal-likelihood shell atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_gpc_log_marginal_likelihood_cached_result,
    witness_gpc_log_marginal_likelihood_mean,
    witness_gpc_log_marginal_likelihood_require_no_multiclass_gradient,
    witness_gpc_log_marginal_likelihood_require_theta_for_gradient,
    witness_gpc_log_marginal_likelihood_theta_shape_message,
    witness_gpc_log_marginal_likelihood_theta_slice,
    witness_gpc_log_marginal_likelihood_use_binary_branch,
    witness_gpc_log_marginal_likelihood_use_compound_theta,
    witness_gpc_log_marginal_likelihood_use_shared_theta,
)


def _bool(value: object) -> bool:
    return isinstance(value, bool)


def _positive_int(value: object) -> bool:
    return bool(isinstance(value, int) and not isinstance(value, bool) and value >= 1)


def _nonnegative_int(value: object) -> bool:
    return bool(isinstance(value, int) and not isinstance(value, bool) and value >= 0)


def _finite_scalar(value: object) -> bool:
    return bool(np.isscalar(value) and not isinstance(value, bool) and np.isfinite(float(value)))


def _finite_vector(values: object) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 1 and array.shape[0] >= 1 and np.all(np.isfinite(array)))


@register_atom(witness_gpc_log_marginal_likelihood_require_theta_for_gradient)
@icontract.require(lambda theta_is_none: _bool(theta_is_none), "theta_is_none must be boolean")
@icontract.require(lambda eval_gradient: _bool(eval_gradient), "eval_gradient must be boolean")
@icontract.ensure(lambda result: result is None, "guard returns None when it does not raise")
def gpc_log_marginal_likelihood_require_theta_for_gradient(
    theta_is_none: bool,
    eval_gradient: bool,
) -> None:
    """Reject gradient evaluation when theta is omitted."""
    if theta_is_none and eval_gradient:
        raise ValueError("Gradient can only be evaluated for theta!=None")


@register_atom(witness_gpc_log_marginal_likelihood_cached_result)
@icontract.require(lambda log_marginal_likelihood_value: _finite_scalar(log_marginal_likelihood_value), "cached log-marginal-likelihood value must be finite")
@icontract.ensure(lambda result, log_marginal_likelihood_value: _finite_scalar(result) and float(result) == float(log_marginal_likelihood_value), "result must preserve the cached log-marginal-likelihood value")
def gpc_log_marginal_likelihood_cached_result(
    log_marginal_likelihood_value: float,
) -> float:
    """Return the cached log-marginal likelihood for theta=None without gradient."""
    return float(log_marginal_likelihood_value)


@register_atom(witness_gpc_log_marginal_likelihood_require_no_multiclass_gradient)
@icontract.require(lambda n_classes: _positive_int(n_classes), "n_classes must be a positive integer")
@icontract.require(lambda eval_gradient: _bool(eval_gradient), "eval_gradient must be boolean")
@icontract.ensure(lambda result: result is None, "guard returns None when it does not raise")
def gpc_log_marginal_likelihood_require_no_multiclass_gradient(
    n_classes: int,
    eval_gradient: bool,
) -> None:
    """Reject gradient evaluation for multiclass GaussianProcessClassifier log-marginal likelihood."""
    if int(n_classes) > 2 and eval_gradient:
        raise NotImplementedError(
            "Gradient of log-marginal-likelihood not implemented for multi-class GPC."
        )


@register_atom(witness_gpc_log_marginal_likelihood_use_binary_branch)
@icontract.require(lambda n_classes: _positive_int(n_classes), "n_classes must be a positive integer")
def gpc_log_marginal_likelihood_use_binary_branch(
    n_classes: int,
) -> bool:
    """Decide whether GaussianProcessClassifier.log_marginal_likelihood uses the binary estimator branch."""
    return int(n_classes) == 2


@register_atom(witness_gpc_log_marginal_likelihood_use_shared_theta)
@icontract.require(lambda theta: _finite_vector(theta), "theta must be a finite nonempty vector")
@icontract.require(lambda n_dims: _positive_int(n_dims), "n_dims must be a positive integer")
@icontract.require(lambda n_classes: _positive_int(n_classes), "n_classes must be a positive integer")
def gpc_log_marginal_likelihood_use_shared_theta(
    theta: NDArray[np.float64],
    n_dims: int,
    n_classes: int,
) -> bool:
    """Decide whether GaussianProcessClassifier.log_marginal_likelihood uses one shared theta for all multiclass sub-kernels."""
    return bool(int(n_classes) > 2 and np.asarray(theta, dtype=np.float64).shape[0] == int(n_dims))


@register_atom(witness_gpc_log_marginal_likelihood_use_compound_theta)
@icontract.require(lambda theta: _finite_vector(theta), "theta must be a finite nonempty vector")
@icontract.require(lambda n_dims: _positive_int(n_dims), "n_dims must be a positive integer")
@icontract.require(lambda n_classes: _positive_int(n_classes), "n_classes must be a positive integer")
def gpc_log_marginal_likelihood_use_compound_theta(
    theta: NDArray[np.float64],
    n_dims: int,
    n_classes: int,
) -> bool:
    """Decide whether GaussianProcessClassifier.log_marginal_likelihood uses a concatenated multiclass theta vector."""
    return bool(int(n_classes) > 2 and np.asarray(theta, dtype=np.float64).shape[0] == int(n_dims) * int(n_classes))


@register_atom(witness_gpc_log_marginal_likelihood_theta_slice)
@icontract.require(lambda theta: _finite_vector(theta), "theta must be a finite nonempty vector")
@icontract.require(lambda n_dims: _positive_int(n_dims), "n_dims must be a positive integer")
@icontract.require(lambda estimator_index: _nonnegative_int(estimator_index), "estimator_index must be a nonnegative integer")
@icontract.ensure(
    lambda result, n_dims: _finite_vector(result) and np.asarray(result, dtype=np.float64).shape == (int(n_dims),),
    "result must be a finite theta slice with length n_dims",
)
def gpc_log_marginal_likelihood_theta_slice(
    theta: NDArray[np.float64],
    n_dims: int,
    estimator_index: int,
) -> NDArray[np.float64]:
    """Extract one multiclass sub-kernel theta block the way GaussianProcessClassifier slices a compound theta."""
    values = np.asarray(theta, dtype=np.float64)
    dims = int(n_dims)
    index = int(estimator_index)
    start = dims * index
    stop = dims * (index + 1)
    result = values[start:stop]
    if result.shape[0] != dims:
        raise ValueError("theta does not contain a full slice for the requested estimator_index")
    return result


@register_atom(witness_gpc_log_marginal_likelihood_mean)
@icontract.require(lambda values: _finite_vector(values), "values must be a finite nonempty vector")
@icontract.ensure(lambda result: _finite_scalar(result), "result must be a finite scalar")
def gpc_log_marginal_likelihood_mean(
    values: NDArray[np.float64],
) -> float:
    """Average multiclass one-vs-rest log-marginal-likelihood callback outputs."""
    return float(np.mean(np.asarray(values, dtype=np.float64)))


@register_atom(witness_gpc_log_marginal_likelihood_theta_shape_message)
@icontract.require(lambda n_dims: _positive_int(n_dims), "n_dims must be a positive integer")
@icontract.require(lambda n_classes: _positive_int(n_classes), "n_classes must be a positive integer")
@icontract.require(lambda theta_size: _positive_int(theta_size), "theta_size must be a positive integer")
@icontract.ensure(
    lambda result, n_dims, n_classes, theta_size: isinstance(result, str)
    and result
    == (
        "Shape of theta must be either %d or %d. Obtained theta with shape %d."
        % (int(n_dims), int(n_dims) * int(n_classes), int(theta_size))
    ),
    "result must match sklearn's invalid-theta-shape message",
)
def gpc_log_marginal_likelihood_theta_shape_message(
    n_dims: int,
    n_classes: int,
    theta_size: int,
) -> str:
    """Format sklearn's invalid theta-shape error message for multiclass GaussianProcessClassifier."""
    return "Shape of theta must be either %d or %d. Obtained theta with shape %d." % (
        int(n_dims),
        int(n_dims) * int(n_classes),
        int(theta_size),
    )
