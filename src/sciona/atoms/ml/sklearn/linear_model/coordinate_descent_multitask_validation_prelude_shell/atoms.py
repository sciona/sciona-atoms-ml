"""Sklearn coordinate-descent multi-output validation prelude atoms."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_cd_multitask_check_x_params,
    witness_cd_multitask_check_y_params,
    witness_cd_multitask_consistent_length_args,
    witness_cd_multitask_shape_counts,
    witness_cd_multitask_validate_data_args,
    witness_cd_multitask_validate_data_kwargs,
    witness_cd_multitask_y_astype_dtype,
)


def _bool(value: object) -> bool:
    return isinstance(value, bool)


def _positive_shape(value: object) -> bool:
    return (
        isinstance(value, tuple)
        and len(value) >= 2
        and all(isinstance(item, (int, np.integer)) and int(item) >= 1 for item in value[:2])
    )


def _dtype(value: object) -> bool:
    try:
        np.dtype(value)
    except (TypeError, ValueError):
        return False
    return True


@register_atom(witness_cd_multitask_check_x_params)
@icontract.require(lambda copy_X: _bool(copy_X), "copy_X must be boolean")
@icontract.require(lambda fit_intercept: _bool(fit_intercept), "fit_intercept must be boolean")
@icontract.ensure(
    lambda result, copy_X, fit_intercept: isinstance(result, dict)
    and result
    == {
        "dtype": [np.float64, np.float32],
        "order": "F",
        "force_writeable": True,
        "copy": copy_X and fit_intercept,
    },
    "X validation params must match MultiTaskElasticNet.fit",
)
def cd_multitask_check_x_params(copy_X: bool, fit_intercept: bool) -> dict[str, object]:
    """Return check_X_params for MultiTaskElasticNet.fit validate_data."""
    return {
        "dtype": [np.float64, np.float32],
        "order": "F",
        "force_writeable": True,
        "copy": copy_X and fit_intercept,
    }


@register_atom(witness_cd_multitask_check_y_params)
@icontract.ensure(
    lambda result, y_context: isinstance(result, dict)
    and result == {"ensure_2d": False, "order": "F"},
    "y validation params must match MultiTaskElasticNet.fit",
)
def cd_multitask_check_y_params(y_context: object) -> dict[str, object]:
    """Return check_y_params for MultiTaskElasticNet.fit validate_data."""
    del y_context
    return {"ensure_2d": False, "order": "F"}


@register_atom(witness_cd_multitask_validate_data_args)
@icontract.ensure(
    lambda result, estimator, X, y: isinstance(result, tuple)
    and len(result) == 3
    and result[0] is estimator
    and result[1] is X
    and result[2] is y,
    "validate_data positional args must preserve estimator, X, and y identity",
)
def cd_multitask_validate_data_args(
    estimator: object, X: object, y: object
) -> tuple[object, object, object]:
    """Return positional args for validate_data(self, X, y, ...)."""
    return (estimator, X, y)


@register_atom(witness_cd_multitask_validate_data_kwargs)
@icontract.require(lambda check_X_params: isinstance(check_X_params, dict), "check_X_params must be a dict")
@icontract.require(lambda check_y_params: isinstance(check_y_params, dict), "check_y_params must be a dict")
@icontract.ensure(
    lambda result, check_X_params, check_y_params: isinstance(result, dict)
    and result == {"validate_separately": (check_X_params, check_y_params)},
    "validate_data kwargs must wrap separate X and y validation params",
)
def cd_multitask_validate_data_kwargs(
    check_X_params: dict[str, object], check_y_params: dict[str, object]
) -> dict[str, object]:
    """Return kwargs for validate_data with separate X and y params."""
    return {"validate_separately": (check_X_params, check_y_params)}


@register_atom(witness_cd_multitask_consistent_length_args)
@icontract.ensure(
    lambda result, X, y: isinstance(result, tuple)
    and len(result) == 2
    and result[0] is X
    and result[1] is y,
    "check_consistent_length args must preserve X and y identity",
)
def cd_multitask_consistent_length_args(X: object, y: object) -> tuple[object, object]:
    """Return positional args for check_consistent_length(X, y)."""
    return (X, y)


@register_atom(witness_cd_multitask_y_astype_dtype)
@icontract.require(lambda y: isinstance(y, np.ndarray), "y must be an ndarray")
@icontract.require(lambda x_dtype: _dtype(x_dtype), "x_dtype must be a NumPy dtype")
@icontract.ensure(
    lambda result, y, x_dtype: isinstance(result, np.ndarray)
    and result.dtype == np.dtype(x_dtype)
    and np.array_equal(result, y.astype(np.dtype(x_dtype))),
    "y dtype cast must match y.astype(X.dtype)",
)
def cd_multitask_y_astype_dtype(
    y: NDArray[np.generic], x_dtype: object
) -> NDArray[np.generic]:
    """Return y cast to the validated X dtype."""
    return y.astype(np.dtype(x_dtype))


@register_atom(witness_cd_multitask_shape_counts)
@icontract.require(lambda x_shape: _positive_shape(x_shape), "x_shape must include positive sample and feature counts")
@icontract.require(lambda y_shape: _positive_shape(y_shape), "y_shape must include positive sample and target counts")
@icontract.require(lambda x_shape, y_shape: int(x_shape[0]) == int(y_shape[0]), "X and y sample counts must match")
@icontract.ensure(
    lambda result, x_shape, y_shape: isinstance(result, tuple)
    and len(result) == 3
    and result == (int(x_shape[0]), int(x_shape[1]), int(y_shape[1])),
    "shape counts must equal X.shape[0], X.shape[1], and y.shape[1]",
)
def cd_multitask_shape_counts(
    x_shape: tuple[int, int], y_shape: tuple[int, int]
) -> tuple[int, int, int]:
    """Return n_samples, n_features, and n_targets from X and y shapes."""
    return (int(x_shape[0]), int(x_shape[1]), int(y_shape[1]))
