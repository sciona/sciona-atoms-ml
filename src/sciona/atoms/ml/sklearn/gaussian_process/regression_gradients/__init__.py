"""Gaussian-process regression log-marginal-likelihood gradient helper atoms."""

from .atoms import (
    gp_log_marginal_gradient,
    gp_log_marginal_gradient_dims,
    gp_log_marginal_gradient_inner_term,
)

__all__ = [
    "gp_log_marginal_gradient",
    "gp_log_marginal_gradient_dims",
    "gp_log_marginal_gradient_inner_term",
]
