"""Gaussian-process classification log-marginal-likelihood shell atoms."""

from .atoms import (
    gpc_log_marginal_likelihood_cached_result,
    gpc_log_marginal_likelihood_mean,
    gpc_log_marginal_likelihood_require_no_multiclass_gradient,
    gpc_log_marginal_likelihood_require_theta_for_gradient,
    gpc_log_marginal_likelihood_theta_shape_message,
    gpc_log_marginal_likelihood_theta_slice,
    gpc_log_marginal_likelihood_use_binary_branch,
    gpc_log_marginal_likelihood_use_compound_theta,
    gpc_log_marginal_likelihood_use_shared_theta,
)

__all__ = [
    "gpc_log_marginal_likelihood_require_theta_for_gradient",
    "gpc_log_marginal_likelihood_cached_result",
    "gpc_log_marginal_likelihood_require_no_multiclass_gradient",
    "gpc_log_marginal_likelihood_use_binary_branch",
    "gpc_log_marginal_likelihood_use_shared_theta",
    "gpc_log_marginal_likelihood_use_compound_theta",
    "gpc_log_marginal_likelihood_theta_slice",
    "gpc_log_marginal_likelihood_mean",
    "gpc_log_marginal_likelihood_theta_shape_message",
]
