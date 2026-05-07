"""Sklearn coordinate-descent estimator validation prelude atoms."""

from __future__ import annotations

import numbers

import icontract
import numpy as np

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_cd_estimator_alpha_zero_warning_message,
    witness_cd_estimator_alpha_zero_warning_required,
    witness_cd_estimator_check_array_y_kwargs,
    witness_cd_estimator_shape_counts,
    witness_cd_estimator_validate_data_args,
    witness_cd_estimator_validate_data_kwargs,
    witness_cd_estimator_x_copied,
)


def _bool(value: object) -> bool:
    return isinstance(value, bool)


def _positive_shape(value: object) -> bool:
    return (
        isinstance(value, tuple)
        and len(value) >= 2
        and all(isinstance(item, (int, np.integer)) and int(item) >= 1 for item in value[:2])
    )


@register_atom(witness_cd_estimator_alpha_zero_warning_required)
@icontract.require(lambda alpha: isinstance(alpha, numbers.Real), "alpha must be a real scalar")
@icontract.ensure(
    lambda result, alpha: _bool(result) and result == (float(alpha) == 0.0),
    "alpha-zero warning predicate must match self.alpha == 0",
)
def cd_estimator_alpha_zero_warning_required(alpha: numbers.Real) -> bool:
    """Return whether ElasticNet.fit emits the alpha-zero convergence warning."""
    return float(alpha) == 0.0


@register_atom(witness_cd_estimator_alpha_zero_warning_message)
@icontract.require(lambda alpha: isinstance(alpha, numbers.Real), "alpha must be a real scalar")
@icontract.ensure(
    lambda result: isinstance(result, str)
    and result
    == (
        "With alpha=0, this algorithm does not converge "
        "well. You are advised to use the LinearRegression "
        "estimator"
    ),
    "alpha-zero warning message must match sklearn formatting",
)
def cd_estimator_alpha_zero_warning_message(alpha: numbers.Real) -> str:
    """Return the warning message used when ElasticNet.fit sees alpha=0."""
    del alpha
    return (
        "With alpha=0, this algorithm does not converge "
        "well. You are advised to use the LinearRegression "
        "estimator"
    )


@register_atom(witness_cd_estimator_x_copied)
@icontract.require(lambda copy_X: _bool(copy_X), "copy_X must be boolean")
@icontract.require(lambda fit_intercept: _bool(fit_intercept), "fit_intercept must be boolean")
@icontract.require(lambda check_input: _bool(check_input), "check_input must be boolean")
@icontract.ensure(
    lambda result, copy_X, fit_intercept, check_input: _bool(result)
    and result == (copy_X and fit_intercept if check_input else False),
    "X_copied must be false unless check_input is true, then copy_X and fit_intercept",
)
def cd_estimator_x_copied(copy_X: bool, fit_intercept: bool, check_input: bool) -> bool:
    """Return ElasticNet.fit X_copied bookkeeping before validate_data."""
    return copy_X and fit_intercept if check_input else False


@register_atom(witness_cd_estimator_validate_data_args)
@icontract.ensure(
    lambda result, estimator, X, y: isinstance(result, tuple)
    and len(result) == 3
    and result[0] is estimator
    and result[1] is X
    and result[2] is y,
    "validate_data positional args must preserve estimator, X, and y identity",
)
def cd_estimator_validate_data_args(estimator: object, X: object, y: object) -> tuple[object, object, object]:
    """Return positional args for validate_data(self, X, y, ...)."""
    return (estimator, X, y)


@register_atom(witness_cd_estimator_validate_data_kwargs)
@icontract.require(lambda x_copied: _bool(x_copied), "x_copied must be boolean")
@icontract.ensure(
    lambda result, x_copied: isinstance(result, dict)
    and result
    == {
        "accept_sparse": "csc",
        "order": "F",
        "dtype": [np.float64, np.float32],
        "force_writeable": True,
        "accept_large_sparse": False,
        "copy": x_copied,
        "multi_output": True,
        "y_numeric": True,
    },
    "validate_data kwargs must match ElasticNet.fit input validation",
)
def cd_estimator_validate_data_kwargs(x_copied: bool) -> dict[str, object]:
    """Return kwargs for ElasticNet.fit validate_data(...)."""
    return {
        "accept_sparse": "csc",
        "order": "F",
        "dtype": [np.float64, np.float32],
        "force_writeable": True,
        "accept_large_sparse": False,
        "copy": x_copied,
        "multi_output": True,
        "y_numeric": True,
    }


@register_atom(witness_cd_estimator_check_array_y_kwargs)
@icontract.require(lambda x_dtype_type: x_dtype_type is not None, "x_dtype_type must be provided")
@icontract.ensure(
    lambda result, x_dtype_type: isinstance(result, dict)
    and result == {"order": "F", "copy": False, "dtype": x_dtype_type, "ensure_2d": False},
    "check_array kwargs must match ElasticNet.fit y validation",
)
def cd_estimator_check_array_y_kwargs(x_dtype_type: object) -> dict[str, object]:
    """Return kwargs for check_array(y, order='F', copy=False, dtype=X.dtype.type, ensure_2d=False)."""
    return {"order": "F", "copy": False, "dtype": x_dtype_type, "ensure_2d": False}


@register_atom(witness_cd_estimator_shape_counts)
@icontract.require(lambda x_shape: _positive_shape(x_shape), "x_shape must include positive sample and feature counts")
@icontract.ensure(
    lambda result, x_shape: isinstance(result, tuple)
    and len(result) == 2
    and result == (int(x_shape[0]), int(x_shape[1])),
    "shape counts must equal the first two entries of X.shape",
)
def cd_estimator_shape_counts(x_shape: tuple[int, int]) -> tuple[int, int]:
    """Return n_samples and n_features extracted from X.shape."""
    return (int(x_shape[0]), int(x_shape[1]))
