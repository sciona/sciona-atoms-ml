"""Gaussian-process regression fit-shell atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_gp_fit_dtype_name,
    witness_gp_fit_stored_train_inputs,
    witness_gp_fit_stored_train_targets,
    witness_gp_fit_use_optimizer_branch,
    witness_gp_fit_validate_ensure_2d,
)


def _bool(value: object) -> bool:
    return isinstance(value, bool)


def _nonnegative_int(value: object) -> bool:
    return bool(isinstance(value, int) and not isinstance(value, bool) and value >= 0)


def _finite_matrix(values: object) -> bool:
    try:
        array = np.asarray(values)
    except (TypeError, ValueError):
        return False
    return bool(
        array.ndim == 2
        and array.shape[0] >= 1
        and array.shape[1] >= 1
        and np.issubdtype(array.dtype, np.number)
        and np.all(np.isfinite(array))
    )


def _finite_vector_or_matrix(values: object) -> bool:
    try:
        array = np.asarray(values)
    except (TypeError, ValueError):
        return False
    return bool(
        array.ndim in {1, 2}
        and array.shape[0] >= 1
        and (array.ndim == 1 or array.shape[1] >= 1)
        and np.issubdtype(array.dtype, np.number)
        and np.all(np.isfinite(array))
    )


def _same_shape_and_values(result: object, source: object) -> bool:
    try:
        left = np.asarray(result)
        right = np.asarray(source)
    except (TypeError, ValueError):
        return False
    return bool(left.shape == right.shape and np.array_equal(left, right))


@register_atom(witness_gp_fit_dtype_name)
@icontract.require(
    lambda kernel_requires_vector_input: _bool(kernel_requires_vector_input),
    "kernel_requires_vector_input must be boolean",
)
@icontract.ensure(
    lambda result: result in {None, "numeric"},
    "dtype mode must match sklearn's fit-time validation choices",
)
def gp_fit_dtype_name(
    kernel_requires_vector_input: bool,
) -> str | None:
    """Resolve sklearn's fit-time validate_data dtype mode for Gaussian-process regression."""
    if kernel_requires_vector_input:
        return "numeric"
    return None


@register_atom(witness_gp_fit_validate_ensure_2d)
@icontract.require(
    lambda kernel_requires_vector_input: _bool(kernel_requires_vector_input),
    "kernel_requires_vector_input must be boolean",
)
@icontract.ensure(lambda result: _bool(result), "ensure_2d mode must be boolean")
def gp_fit_validate_ensure_2d(
    kernel_requires_vector_input: bool,
) -> bool:
    """Resolve sklearn's fit-time validate_data ensure_2d mode for Gaussian-process regression."""
    return bool(kernel_requires_vector_input)


@register_atom(witness_gp_fit_use_optimizer_branch)
@icontract.require(
    lambda optimizer_is_not_none: _bool(optimizer_is_not_none),
    "optimizer_is_not_none must be boolean",
)
@icontract.require(
    lambda kernel_n_dims: _nonnegative_int(kernel_n_dims),
    "kernel_n_dims must be a nonnegative integer",
)
@icontract.ensure(lambda result: _bool(result), "optimizer-branch predicate must be boolean")
def gp_fit_use_optimizer_branch(
    optimizer_is_not_none: bool,
    kernel_n_dims: int,
) -> bool:
    """Decide whether GaussianProcessRegressor.fit enters optimizer selection."""
    return bool(optimizer_is_not_none and kernel_n_dims > 0)


@register_atom(witness_gp_fit_stored_train_inputs)
@icontract.require(lambda X: _finite_matrix(X), "X must be a finite nonempty numeric matrix")
@icontract.require(lambda copy_X_train: _bool(copy_X_train), "copy_X_train must be boolean")
@icontract.ensure(
    lambda result, X: _same_shape_and_values(result, X),
    "stored training inputs must preserve the input matrix values and shape",
)
def gp_fit_stored_train_inputs(
    X: NDArray[np.float64],
    copy_X_train: bool,
) -> NDArray[np.float64]:
    """Store training inputs using sklearn's copy_X_train policy."""
    values = np.asarray(X)
    if copy_X_train:
        return values.copy()
    return values


@register_atom(witness_gp_fit_stored_train_targets)
@icontract.require(lambda y: _finite_vector_or_matrix(y), "y must be a finite nonempty numeric vector or matrix")
@icontract.require(lambda copy_X_train: _bool(copy_X_train), "copy_X_train must be boolean")
@icontract.ensure(
    lambda result, y: _same_shape_and_values(result, y),
    "stored training targets must preserve the target values and shape",
)
def gp_fit_stored_train_targets(
    y: NDArray[np.float64],
    copy_X_train: bool,
) -> NDArray[np.float64]:
    """Store training targets using sklearn's copy_X_train policy."""
    values = np.asarray(y)
    if copy_X_train:
        return values.copy()
    return values
