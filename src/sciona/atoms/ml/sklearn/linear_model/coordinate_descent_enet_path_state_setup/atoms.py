"""Sklearn coordinate-descent enet_path state-setup atoms adapted from scikit-learn."""

from __future__ import annotations

from collections.abc import Sequence

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_cd_enet_path_alpha_count,
    witness_cd_enet_path_coef_path_buffer,
    witness_cd_enet_path_coef_path_shape,
    witness_cd_enet_path_dual_gap_buffer,
    witness_cd_enet_path_initial_coef,
    witness_cd_enet_path_initial_coef_required,
    witness_cd_enet_path_iteration_buffer,
)


def _positive_int(value: object) -> bool:
    return isinstance(value, (int, np.integer)) and int(value) >= 1


def _valid_dtype_name(dtype_name: object) -> bool:
    return isinstance(dtype_name, str) and dtype_name in {"float32", "float64"}


def _valid_coef_shape(shape: object) -> bool:
    return (
        isinstance(shape, tuple)
        and len(shape) in {2, 3}
        and all(_positive_int(dim) for dim in shape)
    )


def _finite_matching_shape(values: object, shape: tuple[int, ...]) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.shape == shape and array.size >= 1 and np.all(np.isfinite(array)))


@register_atom(witness_cd_enet_path_alpha_count)
@icontract.require(
    lambda alphas: isinstance(alphas, Sequence) and len(alphas) >= 1,
    "alphas must be a nonempty sequence",
)
@icontract.ensure(
    lambda result, alphas: _positive_int(result) and int(result) == len(alphas),
    "alpha count must equal len(alphas)",
)
def cd_enet_path_alpha_count(alphas: Sequence[object]) -> int:
    """Return the alpha count used by enet_path."""
    return len(alphas)


@register_atom(witness_cd_enet_path_dual_gap_buffer)
@icontract.require(lambda alpha_count: _positive_int(alpha_count), "alpha_count must be positive")
@icontract.ensure(
    lambda result, alpha_count: isinstance(result, np.ndarray)
    and result.shape == (int(alpha_count),)
    and result.dtype == np.dtype(np.float64),
    "dual-gap buffer must be a one-dimensional float64 array of length alpha_count",
)
def cd_enet_path_dual_gap_buffer(alpha_count: int) -> NDArray[np.float64]:
    """Return the dual-gap buffer allocated by enet_path."""
    return np.empty(int(alpha_count), dtype=np.float64)


@register_atom(witness_cd_enet_path_iteration_buffer)
@icontract.require(lambda alpha_count: _positive_int(alpha_count), "alpha_count must be positive")
@icontract.ensure(lambda result: isinstance(result, list) and result == [], "iteration buffer must start empty")
def cd_enet_path_iteration_buffer(alpha_count: int) -> list[int]:
    """Return the empty iteration-count buffer used by enet_path."""
    del alpha_count
    return []


@register_atom(witness_cd_enet_path_coef_path_shape)
@icontract.require(lambda n_features: _positive_int(n_features), "n_features must be positive")
@icontract.require(lambda alpha_count: _positive_int(alpha_count), "alpha_count must be positive")
@icontract.require(lambda multi_output: isinstance(multi_output, bool), "multi_output must be boolean")
@icontract.require(
    lambda n_targets, multi_output: (n_targets is None and not multi_output)
    or (_positive_int(n_targets) and multi_output),
    "n_targets must be a positive integer exactly in multi-output mode",
)
@icontract.ensure(
    lambda result, n_features, alpha_count, multi_output, n_targets: isinstance(result, tuple)
    and result
    == (
        (int(n_features), int(alpha_count))
        if not multi_output
        else (int(n_targets), int(n_features), int(alpha_count))
    ),
    "coefficient-path shape must match sklearn allocation rules",
)
def cd_enet_path_coef_path_shape(
    n_features: int, alpha_count: int, multi_output: bool, n_targets: int | None
) -> tuple[int, ...]:
    """Return the coefficient-path buffer shape used by enet_path."""
    if not multi_output:
        return (int(n_features), int(alpha_count))
    return (int(n_targets), int(n_features), int(alpha_count))


@register_atom(witness_cd_enet_path_coef_path_buffer)
@icontract.require(lambda coef_shape: _valid_coef_shape(coef_shape), "coef_shape must be a 2D or 3D positive shape")
@icontract.require(lambda dtype_name: _valid_dtype_name(dtype_name), "dtype_name must be float32 or float64")
@icontract.ensure(
    lambda result, coef_shape, dtype_name: isinstance(result, np.ndarray)
    and result.shape == coef_shape
    and result.dtype == np.dtype(dtype_name),
    "coefficient-path buffer must match the requested shape and dtype",
)
def cd_enet_path_coef_path_buffer(
    coef_shape: tuple[int, ...], dtype_name: str
) -> NDArray[np.floating]:
    """Return the coefficient-path buffer allocated by enet_path."""
    return np.empty(coef_shape, dtype=np.dtype(dtype_name))


@register_atom(witness_cd_enet_path_initial_coef_required)
@icontract.ensure(
    lambda result, coef_init: isinstance(result, bool) and result == (coef_init is None),
    "initial-coef predicate must match coef_init is None",
)
def cd_enet_path_initial_coef_required(coef_init: object) -> bool:
    """Return whether enet_path should allocate a zero coefficient buffer."""
    return coef_init is None


@register_atom(witness_cd_enet_path_initial_coef)
@icontract.require(lambda coef_shape: _valid_coef_shape(coef_shape), "coef_shape must be a 2D or 3D positive shape")
@icontract.require(lambda dtype_name: _valid_dtype_name(dtype_name), "dtype_name must be float32 or float64")
@icontract.require(
    lambda coef_shape, coef_init: coef_init is None
    or _finite_matching_shape(coef_init, tuple(int(v) for v in coef_shape[:-1])),
    "coef_init must be None or a finite array matching coef_shape without the alpha axis",
)
@icontract.ensure(
    lambda result, coef_shape, dtype_name: isinstance(result, np.ndarray)
    and result.shape == tuple(int(v) for v in coef_shape[:-1])
    and result.dtype == np.dtype(dtype_name)
    and bool(result.flags["F_CONTIGUOUS"]),
    "initial coefficient buffer must match sklearn shape, dtype, and Fortran order",
)
def cd_enet_path_initial_coef(
    coef_shape: tuple[int, ...], dtype_name: str, coef_init: object
) -> NDArray[np.floating]:
    """Return the initial coefficient buffer used by enet_path."""
    target_shape = tuple(int(v) for v in coef_shape[:-1])
    dtype = np.dtype(dtype_name)
    if coef_init is None:
        return np.zeros(target_shape, dtype=dtype, order="F")
    return np.asfortranarray(coef_init, dtype=dtype)
