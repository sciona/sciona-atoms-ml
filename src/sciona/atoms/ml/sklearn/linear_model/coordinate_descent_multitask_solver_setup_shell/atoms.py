"""Sklearn coordinate-descent multi-output solver setup atoms."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_cd_multitask_coef_fortran_array,
    witness_cd_multitask_fresh_coef_required,
    witness_cd_multitask_initial_coef_zeros,
    witness_cd_multitask_preprocess_data_args,
    witness_cd_multitask_preprocess_data_kwargs,
    witness_cd_multitask_random_state_args,
    witness_cd_multitask_regularization,
    witness_cd_multitask_solver_args,
)


def _bool(value: object) -> bool:
    return isinstance(value, bool)


def _positive_int(value: object) -> bool:
    return isinstance(value, (int, np.integer)) and int(value) >= 1


def _dtype(value: object) -> bool:
    try:
        np.dtype(value)
    except (TypeError, ValueError):
        return False
    return True


def _finite_numeric_array(value: object) -> bool:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        return False
    return bool(array.size >= 1 and np.issubdtype(array.dtype, np.number) and np.all(np.isfinite(array)))


@register_atom(witness_cd_multitask_preprocess_data_args)
@icontract.ensure(
    lambda result, X, y: isinstance(result, tuple)
    and len(result) == 2
    and result[0] is X
    and result[1] is y,
    "_preprocess_data positional args must preserve X and y identity",
)
def cd_multitask_preprocess_data_args(X: object, y: object) -> tuple[object, object]:
    """Return positional args for _preprocess_data(X, y, ...)."""
    return (X, y)


@register_atom(witness_cd_multitask_preprocess_data_kwargs)
@icontract.require(lambda fit_intercept: _bool(fit_intercept), "fit_intercept must be boolean")
@icontract.ensure(
    lambda result, fit_intercept: isinstance(result, dict)
    and result == {"fit_intercept": fit_intercept, "copy": False},
    "_preprocess_data kwargs must match MultiTaskElasticNet.fit",
)
def cd_multitask_preprocess_data_kwargs(fit_intercept: bool) -> dict[str, object]:
    """Return keyword args for _preprocess_data in MultiTaskElasticNet.fit."""
    return {"fit_intercept": fit_intercept, "copy": False}


@register_atom(witness_cd_multitask_fresh_coef_required)
@icontract.require(lambda warm_start: _bool(warm_start), "warm_start must be boolean")
@icontract.require(lambda has_coef_attr: _bool(has_coef_attr), "has_coef_attr must be boolean")
@icontract.ensure(
    lambda result, warm_start, has_coef_attr: _bool(result)
    and result == ((not warm_start) or (not has_coef_attr)),
    "fresh coefficient branch must match not warm_start or missing coef_",
)
def cd_multitask_fresh_coef_required(warm_start: bool, has_coef_attr: bool) -> bool:
    """Return whether MultiTaskElasticNet.fit allocates fresh coefficients."""
    return (not warm_start) or (not has_coef_attr)


@register_atom(witness_cd_multitask_initial_coef_zeros)
@icontract.require(lambda n_targets: _positive_int(n_targets), "n_targets must be positive")
@icontract.require(lambda n_features: _positive_int(n_features), "n_features must be positive")
@icontract.require(lambda dtype: _dtype(dtype), "dtype must be a NumPy dtype")
@icontract.ensure(
    lambda result, n_targets, n_features, dtype: isinstance(result, np.ndarray)
    and result.shape == (int(n_targets), int(n_features))
    and result.dtype == np.dtype(dtype)
    and result.flags["F_CONTIGUOUS"]
    and np.count_nonzero(result) == 0,
    "fresh coefficients must be F-contiguous zeros with expected shape and dtype",
)
def cd_multitask_initial_coef_zeros(
    n_targets: int, n_features: int, dtype: object
) -> NDArray[np.floating]:
    """Return the fresh coefficient matrix allocated before the multitask solver."""
    return np.zeros((int(n_targets), int(n_features)), dtype=np.dtype(dtype), order="F")


@register_atom(witness_cd_multitask_regularization)
@icontract.require(lambda n_samples: _positive_int(n_samples), "n_samples must be positive")
@icontract.ensure(
    lambda result, alpha, l1_ratio, n_samples: isinstance(result, tuple)
    and len(result) == 2
    and np.isclose(float(result[0]), float(alpha) * float(l1_ratio) * int(n_samples))
    and np.isclose(float(result[1]), float(alpha) * (1.0 - float(l1_ratio)) * int(n_samples)),
    "regularization values must match multitask l1_reg and l2_reg scaling",
)
def cd_multitask_regularization(
    alpha: object, l1_ratio: object, n_samples: int
) -> tuple[float, float]:
    """Return l1_reg and l2_reg passed to the compiled multitask solver."""
    return (
        float(alpha) * float(l1_ratio) * int(n_samples),
        float(alpha) * (1.0 - float(l1_ratio)) * int(n_samples),
    )


@register_atom(witness_cd_multitask_coef_fortran_array)
@icontract.require(lambda coef: _finite_numeric_array(coef), "coef must be a finite numeric array")
@icontract.ensure(
    lambda result, coef: isinstance(result, np.ndarray)
    and result.flags["F_CONTIGUOUS"]
    and np.array_equal(result, np.asarray(coef)),
    "coefficient array must preserve values and be Fortran-contiguous",
)
def cd_multitask_coef_fortran_array(coef: object) -> NDArray[np.generic]:
    """Return coef normalized with np.asfortranarray."""
    return np.asfortranarray(coef)


@register_atom(witness_cd_multitask_random_state_args)
@icontract.ensure(
    lambda result, random_state: isinstance(result, tuple)
    and len(result) == 1
    and result[0] is random_state,
    "check_random_state args must preserve random_state identity",
)
def cd_multitask_random_state_args(random_state: object) -> tuple[object]:
    """Return positional args for check_random_state(random_state)."""
    return (random_state,)


@register_atom(witness_cd_multitask_solver_args)
@icontract.require(lambda max_iter: _positive_int(max_iter), "max_iter must be positive")
@icontract.require(lambda random: _bool(random), "random must be boolean")
@icontract.ensure(
    lambda result, coef, l1_reg, l2_reg, X, y, max_iter, tol, checked_random_state, random: isinstance(result, tuple)
    and len(result) == 9
    and result[0] is coef
    and result[1] is l1_reg
    and result[2] is l2_reg
    and result[3] is X
    and result[4] is y
    and result[5] is max_iter
    and result[6] is tol
    and result[7] is checked_random_state
    and result[8] is random,
    "compiled multitask solver args must match sklearn call order",
)
def cd_multitask_solver_args(
    coef: object,
    l1_reg: object,
    l2_reg: object,
    X: object,
    y: object,
    max_iter: int,
    tol: object,
    checked_random_state: object,
    random: bool,
) -> tuple[object, object, object, object, object, int, object, object, bool]:
    """Return positional args for enet_coordinate_descent_multi_task."""
    return (coef, l1_reg, l2_reg, X, y, max_iter, tol, checked_random_state, random)
