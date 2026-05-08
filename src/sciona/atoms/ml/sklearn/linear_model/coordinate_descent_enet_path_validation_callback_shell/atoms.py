"""Sklearn coordinate-descent enet_path validation callback atoms."""

from __future__ import annotations

import icontract
import numpy as np

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_cd_enet_path_check_array_X_args,
    witness_cd_enet_path_check_array_X_kwargs,
    witness_cd_enet_path_check_array_Xy_args,
    witness_cd_enet_path_check_array_Xy_kwargs,
    witness_cd_enet_path_check_array_gram_args,
    witness_cd_enet_path_check_array_gram_kwargs,
    witness_cd_enet_path_check_array_y_args,
    witness_cd_enet_path_check_array_y_kwargs,
)


def _bool(value: object) -> bool:
    return isinstance(value, bool)


def _dtype_type(value: object) -> bool:
    return value in {np.float32, np.float64} or value in {np.dtype("float32").type, np.dtype("float64").type}


@register_atom(witness_cd_enet_path_check_array_X_args)
@icontract.ensure(
    lambda result, X: isinstance(result, tuple) and len(result) == 1 and result[0] is X,
    "X validation args must preserve X identity",
)
def cd_enet_path_check_array_X_args(X: object) -> tuple[object]:
    """Return positional args for check_array(X, ...)."""
    return (X,)


@register_atom(witness_cd_enet_path_check_array_X_kwargs)
@icontract.require(lambda copy_X: _bool(copy_X), "copy_X must be boolean")
@icontract.ensure(
    lambda result, copy_X: isinstance(result, dict)
    and result == {
        "accept_sparse": "csc",
        "dtype": [np.float64, np.float32],
        "order": "F",
        "copy": copy_X,
    },
    "X validation kwargs must match enet_path check_array",
)
def cd_enet_path_check_array_X_kwargs(copy_X: bool) -> dict[str, object]:
    """Return kwargs for enet_path check_array(X, ...)."""
    return {
        "accept_sparse": "csc",
        "dtype": [np.float64, np.float32],
        "order": "F",
        "copy": copy_X,
    }


@register_atom(witness_cd_enet_path_check_array_y_args)
@icontract.ensure(
    lambda result, y: isinstance(result, tuple) and len(result) == 1 and result[0] is y,
    "y validation args must preserve y identity",
)
def cd_enet_path_check_array_y_args(y: object) -> tuple[object]:
    """Return positional args for check_array(y, ...)."""
    return (y,)


@register_atom(witness_cd_enet_path_check_array_y_kwargs)
@icontract.require(lambda x_dtype_type: _dtype_type(x_dtype_type), "x_dtype_type must be float32 or float64 dtype type")
@icontract.ensure(
    lambda result, x_dtype_type: isinstance(result, dict)
    and result == {
        "accept_sparse": "csc",
        "dtype": x_dtype_type,
        "order": "F",
        "copy": False,
        "ensure_2d": False,
    },
    "y validation kwargs must match enet_path check_array",
)
def cd_enet_path_check_array_y_kwargs(x_dtype_type: object) -> dict[str, object]:
    """Return kwargs for enet_path check_array(y, ...)."""
    return {
        "accept_sparse": "csc",
        "dtype": x_dtype_type,
        "order": "F",
        "copy": False,
        "ensure_2d": False,
    }


@register_atom(witness_cd_enet_path_check_array_Xy_args)
@icontract.ensure(
    lambda result, Xy: isinstance(result, tuple) and len(result) == 1 and result[0] is Xy,
    "Xy validation args must preserve Xy identity",
)
def cd_enet_path_check_array_Xy_args(Xy: object) -> tuple[object]:
    """Return positional args for check_array(Xy, ...)."""
    return (Xy,)


@register_atom(witness_cd_enet_path_check_array_Xy_kwargs)
@icontract.require(lambda x_dtype_type: _dtype_type(x_dtype_type), "x_dtype_type must be float32 or float64 dtype type")
@icontract.ensure(
    lambda result, x_dtype_type: isinstance(result, dict)
    and result == {
        "dtype": x_dtype_type,
        "order": "C",
        "copy": False,
        "ensure_2d": False,
    },
    "Xy validation kwargs must match enet_path check_array",
)
def cd_enet_path_check_array_Xy_kwargs(x_dtype_type: object) -> dict[str, object]:
    """Return kwargs for enet_path check_array(Xy, ...)."""
    return {
        "dtype": x_dtype_type,
        "order": "C",
        "copy": False,
        "ensure_2d": False,
    }


@register_atom(witness_cd_enet_path_check_array_gram_args)
@icontract.ensure(
    lambda result, precompute: isinstance(result, tuple)
    and len(result) == 1
    and result[0] is precompute,
    "Gram validation args must preserve precompute identity",
)
def cd_enet_path_check_array_gram_args(precompute: object) -> tuple[object]:
    """Return positional args for check_array(precompute, ...)."""
    return (precompute,)


@register_atom(witness_cd_enet_path_check_array_gram_kwargs)
@icontract.require(lambda x_dtype_type: _dtype_type(x_dtype_type), "x_dtype_type must be float32 or float64 dtype type")
@icontract.ensure(
    lambda result, x_dtype_type: isinstance(result, dict)
    and result == {"dtype": x_dtype_type, "order": "C"},
    "Gram validation kwargs must match enet_path check_array",
)
def cd_enet_path_check_array_gram_kwargs(x_dtype_type: object) -> dict[str, object]:
    """Return kwargs for enet_path check_array(precompute, ...)."""
    return {"dtype": x_dtype_type, "order": "C"}
