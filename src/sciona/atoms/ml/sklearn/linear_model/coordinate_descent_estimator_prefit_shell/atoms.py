"""Sklearn coordinate-descent estimator pre-fit shell atoms."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_cd_estimator_n_targets,
    witness_cd_estimator_pre_fit_args,
    witness_cd_estimator_pre_fit_kwargs,
    witness_cd_estimator_set_order_args,
    witness_cd_estimator_set_order_required,
    witness_cd_estimator_should_copy,
    witness_cd_estimator_xy_column_vector,
    witness_cd_estimator_xy_column_vector_required,
    witness_cd_estimator_y_column_vector,
    witness_cd_estimator_y_column_vector_required,
)


def _bool(value: object) -> bool:
    return isinstance(value, bool)


def _ndim(value: object, ndim: int) -> bool:
    return isinstance(value, np.ndarray) and value.ndim == ndim


def _shape_with_targets(value: object) -> bool:
    return (
        isinstance(value, tuple)
        and len(value) >= 2
        and isinstance(value[1], (int, np.integer))
        and int(value[1]) >= 1
    )


@register_atom(witness_cd_estimator_should_copy)
@icontract.require(lambda copy_X: _bool(copy_X), "copy_X must be boolean")
@icontract.require(lambda x_copied: _bool(x_copied), "x_copied must be boolean")
@icontract.ensure(
    lambda result, copy_X, x_copied: _bool(result) and result == (copy_X and not x_copied),
    "should_copy must match self.copy_X and not X_copied",
)
def cd_estimator_should_copy(copy_X: bool, x_copied: bool) -> bool:
    """Return ElasticNet.fit should_copy before _pre_fit(...)."""
    return copy_X and not x_copied


@register_atom(witness_cd_estimator_pre_fit_args)
@icontract.ensure(
    lambda result, X, y, precompute: isinstance(result, tuple)
    and len(result) == 4
    and result[0] is X
    and result[1] is y
    and result[2] is None
    and result[3] is precompute,
    "_pre_fit positional args must preserve X, y, fixed Xy=None, and precompute identity",
)
def cd_estimator_pre_fit_args(X: object, y: object, precompute: object) -> tuple[object, object, None, object]:
    """Return positional args for _pre_fit(X, y, None, self.precompute, ...)."""
    return (X, y, None, precompute)


@register_atom(witness_cd_estimator_pre_fit_kwargs)
@icontract.require(lambda fit_intercept: _bool(fit_intercept), "fit_intercept must be boolean")
@icontract.require(lambda should_copy: _bool(should_copy), "should_copy must be boolean")
@icontract.require(lambda check_input: _bool(check_input), "check_input must be boolean")
@icontract.ensure(
    lambda result, fit_intercept, should_copy, check_input, sample_weight: isinstance(result, dict)
    and result == {
        "fit_intercept": fit_intercept,
        "copy": should_copy,
        "check_input": check_input,
        "sample_weight": sample_weight,
    },
    "_pre_fit kwargs must match ElasticNet.fit setup",
)
def cd_estimator_pre_fit_kwargs(
    fit_intercept: bool, should_copy: bool, check_input: bool, sample_weight: object
) -> dict[str, object]:
    """Return kwargs for _pre_fit(...)."""
    return {
        "fit_intercept": fit_intercept,
        "copy": should_copy,
        "check_input": check_input,
        "sample_weight": sample_weight,
    }


@register_atom(witness_cd_estimator_set_order_required)
@icontract.require(lambda check_input: _bool(check_input), "check_input must be boolean")
@icontract.ensure(
    lambda result, check_input, sample_weight: _bool(result)
    and result == (check_input or sample_weight is not None),
    "_set_order branch must match check_input or sample_weight is not None",
)
def cd_estimator_set_order_required(check_input: bool, sample_weight: object) -> bool:
    """Return whether ElasticNet.fit calls _set_order(X, y, order='F')."""
    return check_input or sample_weight is not None


@register_atom(witness_cd_estimator_set_order_args)
@icontract.ensure(
    lambda result, X, y: isinstance(result, tuple)
    and len(result) == 2
    and result[0] is X
    and result[1] is y,
    "_set_order positional args must preserve X and y identity",
)
def cd_estimator_set_order_args(X: object, y: object) -> tuple[object, object]:
    """Return positional args for _set_order(X, y, order='F')."""
    return (X, y)


@register_atom(witness_cd_estimator_y_column_vector_required)
@icontract.require(lambda y_ndim: isinstance(y_ndim, (int, np.integer)) and int(y_ndim) >= 1, "y_ndim must be positive")
@icontract.ensure(
    lambda result, y_ndim: _bool(result) and result == (int(y_ndim) == 1),
    "y column-vector branch must match y.ndim == 1",
)
def cd_estimator_y_column_vector_required(y_ndim: int) -> bool:
    """Return whether ElasticNet.fit expands y with y[:, np.newaxis]."""
    return int(y_ndim) == 1


@register_atom(witness_cd_estimator_y_column_vector)
@icontract.require(lambda y: _ndim(y, 1), "y must be a one-dimensional ndarray")
@icontract.ensure(
    lambda result, y: isinstance(result, np.ndarray)
    and result.shape == (y.shape[0], 1)
    and np.array_equal(result[:, 0], y),
    "y column-vector normalization must match y[:, np.newaxis]",
)
def cd_estimator_y_column_vector(y: NDArray[np.floating]) -> NDArray[np.floating]:
    """Return y[:, np.newaxis] for mono-output estimator fitting."""
    return y[:, np.newaxis]


@register_atom(witness_cd_estimator_xy_column_vector_required)
@icontract.ensure(
    lambda result, Xy: _bool(result)
    and result == (Xy is not None and isinstance(Xy, np.ndarray) and Xy.ndim == 1),
    "Xy column-vector branch must match Xy is not None and Xy.ndim == 1",
)
def cd_estimator_xy_column_vector_required(Xy: object) -> bool:
    """Return whether ElasticNet.fit expands Xy with Xy[:, np.newaxis]."""
    return Xy is not None and isinstance(Xy, np.ndarray) and Xy.ndim == 1


@register_atom(witness_cd_estimator_xy_column_vector)
@icontract.require(lambda Xy: _ndim(Xy, 1), "Xy must be a one-dimensional ndarray")
@icontract.ensure(
    lambda result, Xy: isinstance(result, np.ndarray)
    and result.shape == (Xy.shape[0], 1)
    and np.array_equal(result[:, 0], Xy),
    "Xy column-vector normalization must match Xy[:, np.newaxis]",
)
def cd_estimator_xy_column_vector(Xy: NDArray[np.floating]) -> NDArray[np.floating]:
    """Return Xy[:, np.newaxis] for mono-output precomputed Xy."""
    return Xy[:, np.newaxis]


@register_atom(witness_cd_estimator_n_targets)
@icontract.require(lambda y_shape: _shape_with_targets(y_shape), "y_shape must include a positive target count")
@icontract.ensure(
    lambda result, y_shape: isinstance(result, int) and result == int(y_shape[1]),
    "n_targets must equal y.shape[1]",
)
def cd_estimator_n_targets(y_shape: tuple[int, int]) -> int:
    """Return n_targets extracted from y.shape[1]."""
    return int(y_shape[1])
