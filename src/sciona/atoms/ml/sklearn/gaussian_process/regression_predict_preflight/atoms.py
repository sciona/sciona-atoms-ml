"""Gaussian-process regression predict preflight helper atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_gp_predict_dtype_name,
    witness_gp_predict_require_single_uncertainty_mode,
    witness_gp_predict_use_prior_branch,
    witness_gp_predict_validate_ensure_2d,
)


def _bool(value: object) -> bool:
    return isinstance(value, bool)


@register_atom(witness_gp_predict_require_single_uncertainty_mode)
@icontract.require(lambda return_std: _bool(return_std), "return_std must be boolean")
@icontract.require(lambda return_cov: _bool(return_cov), "return_cov must be boolean")
@icontract.ensure(lambda result: result is None, "uncertainty-mode guard returns None when it does not raise")
def gp_predict_require_single_uncertainty_mode(
    return_std: bool,
    return_cov: bool,
) -> None:
    """Reject requesting both predictive standard deviation and covariance."""
    if return_std and return_cov:
        raise RuntimeError("At most one of return_std or return_cov can be requested.")


@register_atom(witness_gp_predict_dtype_name)
@icontract.require(lambda kernel_is_none: _bool(kernel_is_none), "kernel_is_none must be boolean")
@icontract.require(lambda kernel_requires_vector_input: _bool(kernel_requires_vector_input), "kernel_requires_vector_input must be boolean")
@icontract.ensure(lambda result: result in {None, "numeric"}, "dtype mode must match sklearn's validation choices")
def gp_predict_dtype_name(
    kernel_is_none: bool,
    kernel_requires_vector_input: bool,
) -> str | None:
    """Resolve sklearn's predict-time validate_data dtype mode."""
    if kernel_is_none or kernel_requires_vector_input:
        return "numeric"
    return None


@register_atom(witness_gp_predict_validate_ensure_2d)
@icontract.require(lambda kernel_is_none: _bool(kernel_is_none), "kernel_is_none must be boolean")
@icontract.require(lambda kernel_requires_vector_input: _bool(kernel_requires_vector_input), "kernel_requires_vector_input must be boolean")
@icontract.ensure(lambda result: _bool(result), "ensure_2d mode must be boolean")
def gp_predict_validate_ensure_2d(
    kernel_is_none: bool,
    kernel_requires_vector_input: bool,
) -> bool:
    """Resolve sklearn's predict-time validate_data ensure_2d mode."""
    return bool(kernel_is_none or kernel_requires_vector_input)


@register_atom(witness_gp_predict_use_prior_branch)
@icontract.require(lambda has_x_train: _bool(has_x_train), "has_x_train must be boolean")
@icontract.ensure(lambda result: _bool(result), "prior-branch predicate must be boolean")
def gp_predict_use_prior_branch(
    has_x_train: bool,
) -> bool:
    """Decide whether GaussianProcessRegressor.predict uses the unfitted prior branch."""
    return not has_x_train
