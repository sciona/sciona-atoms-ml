"""Gaussian-process classification fit-state shell atoms."""

from .atoms import (
    gpc_fit_binary_base_estimator,
    gpc_fit_binary_log_marginal_likelihood_value,
    gpc_fit_multiclass_log_marginal_likelihood_value,
    gpc_fit_one_vs_one_estimator,
    gpc_fit_one_vs_rest_estimator,
    gpc_fit_return_self,
)

__all__ = [
    "gpc_fit_binary_base_estimator",
    "gpc_fit_one_vs_rest_estimator",
    "gpc_fit_one_vs_one_estimator",
    "gpc_fit_binary_log_marginal_likelihood_value",
    "gpc_fit_multiclass_log_marginal_likelihood_value",
    "gpc_fit_return_self",
]
