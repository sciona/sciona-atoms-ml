"""Binary Gaussian-process classification post-fit attribute atoms."""

from .atoms import (
    gpc_binary_fit_L,
    gpc_binary_fit_log_marginal_likelihood_value,
    gpc_binary_fit_pi,
    gpc_binary_fit_return_self,
    gpc_binary_fit_W_sr,
)

__all__ = [
    "gpc_binary_fit_pi",
    "gpc_binary_fit_W_sr",
    "gpc_binary_fit_L",
    "gpc_binary_fit_log_marginal_likelihood_value",
    "gpc_binary_fit_return_self",
]
