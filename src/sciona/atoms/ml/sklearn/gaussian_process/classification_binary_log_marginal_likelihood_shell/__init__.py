"""Binary Gaussian-process classification log-marginal-likelihood shell atoms adapted from scikit-learn."""

from .atoms import (
    gpc_binary_log_marginal_likelihood_cached_result,
    gpc_binary_log_marginal_likelihood_kernel,
    gpc_binary_log_marginal_likelihood_require_theta_for_gradient,
    gpc_binary_log_marginal_likelihood_result,
    gpc_binary_log_marginal_likelihood_use_gradient_branch,
)

__all__ = [
    "gpc_binary_log_marginal_likelihood_cached_result",
    "gpc_binary_log_marginal_likelihood_kernel",
    "gpc_binary_log_marginal_likelihood_require_theta_for_gradient",
    "gpc_binary_log_marginal_likelihood_result",
    "gpc_binary_log_marginal_likelihood_use_gradient_branch",
]
