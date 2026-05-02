"""Gaussian-process regression log-marginal-likelihood shell atoms adapted from scikit-learn."""

from .atoms import (
    gp_log_marginal_likelihood_cached_result,
    gp_log_marginal_likelihood_cholesky_failure_result,
    gp_log_marginal_likelihood_kernel,
    gp_log_marginal_likelihood_require_theta_for_gradient,
    gp_log_marginal_likelihood_train_targets,
)

__all__ = [
    "gp_log_marginal_likelihood_cached_result",
    "gp_log_marginal_likelihood_cholesky_failure_result",
    "gp_log_marginal_likelihood_kernel",
    "gp_log_marginal_likelihood_require_theta_for_gradient",
    "gp_log_marginal_likelihood_train_targets",
]
