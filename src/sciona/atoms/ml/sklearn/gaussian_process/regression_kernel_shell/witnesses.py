"""Ghost witnesses for Gaussian-process regression kernel-shell atoms."""

from __future__ import annotations

from sklearn.gaussian_process.kernels import Kernel


def witness_gp_fit_kernel(kernel: Kernel | None) -> Kernel:
    """Describe the fit-time kernel object used by GaussianProcessRegressor."""
    del kernel
    raise NotImplementedError


def witness_gp_predict_prior_kernel(kernel: Kernel | None) -> Kernel:
    """Describe the prior-prediction kernel object used by GaussianProcessRegressor."""
    del kernel
    raise NotImplementedError


def witness_gp_regression_requires_fit_tag(parent_requires_fit: bool) -> bool:
    """Describe the requires_fit tag override for GaussianProcessRegressor."""
    del parent_requires_fit
    return False
