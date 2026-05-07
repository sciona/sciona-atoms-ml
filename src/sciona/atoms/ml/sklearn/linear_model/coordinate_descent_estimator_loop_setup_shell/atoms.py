"""Sklearn coordinate-descent estimator loop setup atoms."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_cd_estimator_dual_gaps_zeros,
    witness_cd_estimator_initial_coef_required,
    witness_cd_estimator_initial_coef_zeros,
    witness_cd_estimator_loop_this_xy,
    witness_cd_estimator_n_iter_list_initial,
    witness_cd_estimator_path_args,
    witness_cd_estimator_path_kwargs,
    witness_cd_estimator_single_alpha_grid,
    witness_cd_estimator_warm_start_coef_matrix,
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


def _path_kwargs_match(
    result: object,
    l1_ratio: object,
    alpha: object,
    precompute: object,
    this_Xy: object,
    coef_init: object,
    positive: bool,
    tol: object,
    X_offset: object,
    X_scale: object,
    max_iter: int,
    random_state: object,
    selection: str,
    sample_weight: object,
) -> bool:
    if not isinstance(result, dict):
        return False
    expected_keys = {
        "l1_ratio",
        "eps",
        "n_alphas",
        "alphas",
        "precompute",
        "Xy",
        "copy_X",
        "coef_init",
        "verbose",
        "return_n_iter",
        "positive",
        "check_input",
        "tol",
        "X_offset",
        "X_scale",
        "max_iter",
        "random_state",
        "selection",
        "sample_weight",
    }
    return (
        set(result) == expected_keys
        and result["l1_ratio"] is l1_ratio
        and result["eps"] is None
        and result["n_alphas"] is None
        and isinstance(result["alphas"], list)
        and len(result["alphas"]) == 1
        and result["alphas"][0] is alpha
        and result["precompute"] is precompute
        and result["Xy"] is this_Xy
        and result["copy_X"] is True
        and result["coef_init"] is coef_init
        and result["verbose"] is False
        and result["return_n_iter"] is True
        and result["positive"] is positive
        and result["check_input"] is False
        and result["tol"] is tol
        and result["X_offset"] is X_offset
        and result["X_scale"] is X_scale
        and result["max_iter"] is max_iter
        and result["random_state"] is random_state
        and result["selection"] is selection
        and result["sample_weight"] is sample_weight
    )


@register_atom(witness_cd_estimator_initial_coef_required)
@icontract.require(lambda warm_start: _bool(warm_start), "warm_start must be boolean")
@icontract.require(lambda has_coef_attr: _bool(has_coef_attr), "has_coef_attr must be boolean")
@icontract.ensure(
    lambda result, warm_start, has_coef_attr: _bool(result)
    and result == ((not warm_start) or (not has_coef_attr)),
    "fresh coefficient branch must match not warm_start or missing coef_",
)
def cd_estimator_initial_coef_required(warm_start: bool, has_coef_attr: bool) -> bool:
    """Return whether ElasticNet.fit allocates a fresh coefficient matrix."""
    return (not warm_start) or (not has_coef_attr)


@register_atom(witness_cd_estimator_initial_coef_zeros)
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
def cd_estimator_initial_coef_zeros(
    n_targets: int, n_features: int, dtype: object
) -> NDArray[np.floating]:
    """Return the fresh coefficient matrix allocated by ElasticNet.fit."""
    return np.zeros((int(n_targets), int(n_features)), dtype=np.dtype(dtype), order="F")


@register_atom(witness_cd_estimator_warm_start_coef_matrix)
@icontract.require(lambda coef: isinstance(coef, np.ndarray), "coef must be an ndarray")
@icontract.require(lambda coef: coef.ndim in {1, 2}, "coef must be one- or two-dimensional")
@icontract.ensure(
    lambda result, coef: isinstance(result, np.ndarray)
    and (
        (coef.ndim == 1 and result.shape == (1, coef.shape[0]) and np.array_equal(result[0], coef))
        or (coef.ndim == 2 and result is coef)
    ),
    "warm-start coefficients must preserve matrix inputs or expand vectors with np.newaxis",
)
def cd_estimator_warm_start_coef_matrix(coef: NDArray[np.floating]) -> NDArray[np.floating]:
    """Return warm-start coef_ normalized to two dimensions."""
    return coef[np.newaxis, :] if coef.ndim == 1 else coef


@register_atom(witness_cd_estimator_dual_gaps_zeros)
@icontract.require(lambda n_targets: _positive_int(n_targets), "n_targets must be positive")
@icontract.require(lambda dtype: _dtype(dtype), "dtype must be a NumPy dtype")
@icontract.ensure(
    lambda result, n_targets, dtype: isinstance(result, np.ndarray)
    and result.shape == (int(n_targets),)
    and result.dtype == np.dtype(dtype)
    and np.count_nonzero(result) == 0,
    "dual-gap buffer must be zeros with expected shape and dtype",
)
def cd_estimator_dual_gaps_zeros(n_targets: int, dtype: object) -> NDArray[np.floating]:
    """Return the dual-gap buffer allocated before the target loop."""
    return np.zeros(int(n_targets), dtype=np.dtype(dtype))


@register_atom(witness_cd_estimator_n_iter_list_initial)
@icontract.require(lambda n_targets: _positive_int(n_targets), "n_targets must be positive")
@icontract.ensure(lambda result: isinstance(result, list) and result == [], "n_iter_ must start as an empty list")
def cd_estimator_n_iter_list_initial(n_targets: int) -> list[int]:
    """Return the empty n_iter_ list assigned before the target loop."""
    del n_targets
    return []


@register_atom(witness_cd_estimator_loop_this_xy)
@icontract.require(lambda target_index: isinstance(target_index, (int, np.integer)) and int(target_index) >= 0, "target_index must be nonnegative")
@icontract.require(lambda Xy, target_index: Xy is None or (isinstance(Xy, np.ndarray) and Xy.ndim == 2 and int(target_index) < Xy.shape[1]), "Xy must be None or a 2D array with target_index in range")
@icontract.ensure(
    lambda result, Xy, target_index: (
        result is None
        if Xy is None
        else np.array_equal(result, Xy[:, int(target_index)])
    ),
    "per-target Xy must be None or Xy[:, k]",
)
def cd_estimator_loop_this_xy(Xy: object, target_index: int) -> object:
    """Return this_Xy for one target in ElasticNet.fit."""
    return None if Xy is None else Xy[:, int(target_index)]


@register_atom(witness_cd_estimator_single_alpha_grid)
@icontract.ensure(
    lambda result, alpha: isinstance(result, list) and len(result) == 1 and result[0] is alpha,
    "path alphas payload must be a one-element list preserving alpha identity",
)
def cd_estimator_single_alpha_grid(alpha: object) -> list[object]:
    """Return the one-alpha list passed into self.path(...)."""
    return [alpha]


@register_atom(witness_cd_estimator_path_args)
@icontract.require(lambda target_index: isinstance(target_index, (int, np.integer)) and int(target_index) >= 0, "target_index must be nonnegative")
@icontract.require(lambda y, target_index: isinstance(y, np.ndarray) and y.ndim == 2 and int(target_index) < y.shape[1], "y must be 2D with target_index in range")
@icontract.ensure(
    lambda result, X, y, target_index: isinstance(result, tuple)
    and len(result) == 2
    and result[0] is X
    and np.array_equal(result[1], y[:, int(target_index)]),
    "path positional args must preserve X and select y[:, k]",
)
def cd_estimator_path_args(X: object, y: NDArray[np.floating], target_index: int) -> tuple[object, NDArray[np.floating]]:
    """Return positional args for one self.path(X, y[:, k], ...) callback."""
    return (X, y[:, int(target_index)])


@register_atom(witness_cd_estimator_path_kwargs)
@icontract.require(lambda positive: _bool(positive), "positive must be boolean")
@icontract.require(lambda max_iter: _positive_int(max_iter), "max_iter must be positive")
@icontract.require(lambda selection: isinstance(selection, str), "selection must be a string")
@icontract.ensure(
    lambda result, l1_ratio, alpha, precompute, this_Xy, coef_init, positive, tol, X_offset, X_scale, max_iter, random_state, selection, sample_weight: _path_kwargs_match(
        result,
        l1_ratio,
        alpha,
        precompute,
        this_Xy,
        coef_init,
        positive,
        tol,
        X_offset,
        X_scale,
        max_iter,
        random_state,
        selection,
        sample_weight,
    ),
    "path kwargs must match ElasticNet.fit target-loop callback setup",
)
def cd_estimator_path_kwargs(
    l1_ratio: object,
    alpha: object,
    precompute: object,
    this_Xy: object,
    coef_init: object,
    positive: bool,
    tol: object,
    X_offset: object,
    X_scale: object,
    max_iter: int,
    random_state: object,
    selection: str,
    sample_weight: object,
) -> dict[str, object]:
    """Return kwargs for one self.path(...) callback in ElasticNet.fit."""
    return {
        "l1_ratio": l1_ratio,
        "eps": None,
        "n_alphas": None,
        "alphas": [alpha],
        "precompute": precompute,
        "Xy": this_Xy,
        "copy_X": True,
        "coef_init": coef_init,
        "verbose": False,
        "return_n_iter": True,
        "positive": positive,
        "check_input": False,
        "tol": tol,
        "X_offset": X_offset,
        "X_scale": X_scale,
        "max_iter": max_iter,
        "random_state": random_state,
        "selection": selection,
        "sample_weight": sample_weight,
    }
