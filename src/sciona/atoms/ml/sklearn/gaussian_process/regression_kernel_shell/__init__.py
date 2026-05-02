"""Gaussian-process regression kernel-shell atoms adapted from scikit-learn."""

from .atoms import gp_fit_kernel, gp_predict_prior_kernel, gp_regression_requires_fit_tag

__all__ = [
    "gp_fit_kernel",
    "gp_predict_prior_kernel",
    "gp_regression_requires_fit_tag",
]
