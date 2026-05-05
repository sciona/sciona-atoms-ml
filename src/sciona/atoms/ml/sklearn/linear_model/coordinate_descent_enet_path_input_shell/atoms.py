"""Sklearn coordinate-descent enet_path input-shell atoms adapted from scikit-learn."""

from __future__ import annotations

from collections.abc import Mapping

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_cd_enet_path_alpha_grid_required,
    witness_cd_enet_path_check_input_branch,
    witness_cd_enet_path_prefit_kwargs,
    witness_cd_enet_path_sparse_scaling,
    witness_cd_enet_path_sparse_scaling_required,
    witness_cd_enet_path_unexpected_params_guard_required,
    witness_cd_enet_path_Xy_validation_required,
)


def _positive_int(value: object) -> bool:
    return isinstance(value, (int, np.integer)) and int(value) >= 1


def _valid_dtype_name(dtype_name: object) -> bool:
    return isinstance(dtype_name, str) and dtype_name in {"float32", "float64"}


def _finite_vector_with_length(values: object, length: int) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 1 and array.shape[0] == length and np.all(np.isfinite(array)))


def _finite_nonzero_vector_with_length(values: object, length: int) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(
        array.ndim == 1
        and array.shape[0] == length
        and np.all(np.isfinite(array))
        and np.all(array != 0.0)
    )


@register_atom(witness_cd_enet_path_unexpected_params_guard_required)
@icontract.require(lambda params: isinstance(params, Mapping), "params must be a mapping")
@icontract.ensure(
    lambda result, params: isinstance(result, bool) and result == (len(params) > 0),
    "guard predicate must match len(params) > 0",
)
def cd_enet_path_unexpected_params_guard_required(params: Mapping[object, object]) -> bool:
    """Return whether enet_path should raise on leftover params."""
    return len(params) > 0


@register_atom(witness_cd_enet_path_check_input_branch)
@icontract.require(lambda check_input: isinstance(check_input, bool), "check_input must be boolean")
@icontract.ensure(
    lambda result, check_input: isinstance(result, bool) and result == check_input,
    "branch predicate must match check_input",
)
def cd_enet_path_check_input_branch(check_input: bool) -> bool:
    """Return whether enet_path should run input validation."""
    return check_input


@register_atom(witness_cd_enet_path_Xy_validation_required)
@icontract.ensure(
    lambda result, Xy: isinstance(result, bool) and result == (Xy is not None),
    "Xy validation predicate must match Xy is not None",
)
def cd_enet_path_Xy_validation_required(Xy: object) -> bool:
    """Return whether enet_path should validate Xy."""
    return Xy is not None


@register_atom(witness_cd_enet_path_sparse_scaling_required)
@icontract.require(lambda multi_output: isinstance(multi_output, bool), "multi_output must be boolean")
@icontract.require(lambda x_is_sparse: isinstance(x_is_sparse, bool), "x_is_sparse must be boolean")
@icontract.ensure(
    lambda result, multi_output, x_is_sparse: isinstance(result, bool)
    and result == ((not multi_output) and x_is_sparse),
    "sparse-scaling predicate must match the sparse mono-output branch",
)
def cd_enet_path_sparse_scaling_required(multi_output: bool, x_is_sparse: bool) -> bool:
    """Return whether enet_path should build X_sparse_scaling."""
    return (not multi_output) and x_is_sparse


@register_atom(witness_cd_enet_path_sparse_scaling)
@icontract.require(lambda n_features: _positive_int(n_features), "n_features must be a positive integer")
@icontract.require(lambda dtype_name: _valid_dtype_name(dtype_name), "dtype_name must be float32 or float64")
@icontract.require(
    lambda X_offset_param, n_features: X_offset_param is None
    or _finite_vector_with_length(X_offset_param, int(n_features)),
    "X_offset_param must be None or a finite vector of length n_features",
)
@icontract.require(
    lambda X_offset_param, X_scale_param, n_features: X_offset_param is None
    or _finite_nonzero_vector_with_length(X_scale_param, int(n_features)),
    "X_scale_param must be a finite nonzero vector of length n_features when offsets are supplied",
)
@icontract.ensure(
    lambda result, n_features, dtype_name: isinstance(result, np.ndarray)
    and result.shape == (int(n_features),)
    and result.dtype == np.dtype(dtype_name),
    "sparse scaling must be a one-dimensional array of the requested dtype",
)
def cd_enet_path_sparse_scaling(
    X_offset_param: object,
    X_scale_param: object,
    n_features: int,
    dtype_name: str,
) -> NDArray[np.floating]:
    """Return the sparse scaling vector used by enet_path."""
    dtype = np.dtype(dtype_name)
    if X_offset_param is None:
        return np.zeros(int(n_features), dtype=dtype)
    scaling = np.asarray(X_offset_param, dtype=np.float64) / np.asarray(X_scale_param, dtype=np.float64)
    return np.asarray(scaling, dtype=dtype)


@register_atom(witness_cd_enet_path_prefit_kwargs)
@icontract.require(lambda check_input: isinstance(check_input, bool), "check_input must be boolean")
@icontract.ensure(
    lambda result, check_input: result
    == {"fit_intercept": False, "copy": False, "check_input": check_input},
    "pre-fit kwargs must match the fixed enet_path shell",
)
def cd_enet_path_prefit_kwargs(check_input: bool) -> dict[str, object]:
    """Return the fixed _pre_fit keyword arguments used by enet_path."""
    return {"fit_intercept": False, "copy": False, "check_input": check_input}


@register_atom(witness_cd_enet_path_alpha_grid_required)
@icontract.ensure(
    lambda result, alphas: isinstance(result, bool) and result == (alphas is None),
    "alpha-grid predicate must match alphas is None",
)
def cd_enet_path_alpha_grid_required(alphas: object) -> bool:
    """Return whether enet_path should call _alpha_grid."""
    return alphas is None
