"""Gaussian-process regression kernel-shell atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
from sklearn.base import clone
from sklearn.gaussian_process.kernels import (
    ConstantKernel as C,
    Kernel,
    RBF,
)

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_gp_fit_kernel,
    witness_gp_predict_prior_kernel,
    witness_gp_regression_requires_fit_tag,
)


def _kernel_or_none(value: object) -> bool:
    return value is None or isinstance(value, Kernel)


def _kernel(value: object) -> bool:
    return isinstance(value, Kernel)


def _bool(value: object) -> bool:
    return isinstance(value, bool)


def _is_default_kernel(value: object) -> bool:
    if not isinstance(value, Kernel):
        return False
    return bool(repr(value) == repr(C(1.0, constant_value_bounds="fixed") * RBF(1.0, length_scale_bounds="fixed")))


def _default_kernel() -> Kernel:
    return C(1.0, constant_value_bounds="fixed") * RBF(1.0, length_scale_bounds="fixed")


@register_atom(witness_gp_fit_kernel)
@icontract.require(lambda kernel=None: _kernel_or_none(kernel), "kernel must be None or a sklearn kernel instance")
@icontract.ensure(lambda result: _kernel(result), "fit kernel must be a sklearn kernel instance")
@icontract.ensure(
    lambda result, kernel=None: _is_default_kernel(result) if kernel is None else (repr(result) == repr(kernel) and result is not kernel),
    "fit kernel must be the default kernel when kernel is None, or a clone of the supplied kernel otherwise",
)
def gp_fit_kernel(kernel: Kernel | None = None) -> Kernel:
    """Resolve GaussianProcessRegressor.fit's kernel object before optimization."""
    if kernel is None:
        return _default_kernel()
    return clone(kernel)


@register_atom(witness_gp_predict_prior_kernel)
@icontract.require(lambda kernel=None: _kernel_or_none(kernel), "kernel must be None or a sklearn kernel instance")
@icontract.ensure(lambda result: _kernel(result), "prior-predict kernel must be a sklearn kernel instance")
@icontract.ensure(
    lambda result, kernel=None: _is_default_kernel(result) if kernel is None else result is kernel,
    "prior-predict kernel must be the default kernel when kernel is None, or the supplied kernel object otherwise",
)
def gp_predict_prior_kernel(kernel: Kernel | None = None) -> Kernel:
    """Resolve GaussianProcessRegressor.predict's unfitted prior kernel object."""
    if kernel is None:
        return _default_kernel()
    return kernel


@register_atom(witness_gp_regression_requires_fit_tag)
@icontract.require(lambda parent_requires_fit: _bool(parent_requires_fit), "parent_requires_fit must be boolean")
@icontract.ensure(lambda result: _bool(result) and result is False, "GaussianProcessRegressor.requires_fit must be False")
def gp_regression_requires_fit_tag(parent_requires_fit: bool) -> bool:
    """Override the requires_fit tag for GaussianProcessRegressor."""
    del parent_requires_fit
    return False
