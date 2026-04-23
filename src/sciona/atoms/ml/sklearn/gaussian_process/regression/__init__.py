"""Estimator-independent Gaussian-process regression linear-algebra atoms."""

from .atoms import (
    gp_dual_coefficients,
    gp_log_marginal_likelihood,
    gp_posterior_cross_solve,
    gp_posterior_predictive_covariance,
    gp_posterior_predictive_mean,
    gp_posterior_predictive_std,
    gp_regularized_train_kernel,
    gp_train_cholesky,
)

__all__ = [
    "gp_dual_coefficients",
    "gp_log_marginal_likelihood",
    "gp_posterior_cross_solve",
    "gp_posterior_predictive_covariance",
    "gp_posterior_predictive_mean",
    "gp_posterior_predictive_std",
    "gp_regularized_train_kernel",
    "gp_train_cholesky",
]
