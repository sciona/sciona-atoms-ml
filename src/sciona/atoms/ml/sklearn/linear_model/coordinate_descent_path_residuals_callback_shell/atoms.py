"""Sklearn coordinate-descent path-residual callback-shell atoms adapted from scikit-learn."""

from __future__ import annotations

from collections.abc import Sequence

import icontract

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_cd_path_residuals_check_array_accept_sparse,
    witness_cd_path_residuals_check_array_dtype,
    witness_cd_path_residuals_check_array_order,
    witness_cd_path_residuals_path_result_alphas,
    witness_cd_path_residuals_path_result_coefs,
)


def _path_result_valid(value: object) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) == 3


@register_atom(witness_cd_path_residuals_check_array_accept_sparse)
@icontract.ensure(
    lambda result: isinstance(result, str) and result == "csc",
    "check_array accept_sparse kwarg must be fixed to 'csc'",
)
def cd_path_residuals_check_array_accept_sparse(dtype: object) -> str:
    """Return the fixed accept_sparse kwarg passed to check_array by _path_residuals."""
    del dtype
    return "csc"


@register_atom(witness_cd_path_residuals_check_array_dtype)
@icontract.ensure(
    lambda result, dtype: result is dtype,
    "check_array dtype kwarg must preserve the supplied dtype object",
)
def cd_path_residuals_check_array_dtype(dtype: object) -> object:
    """Return the dtype kwarg passed to check_array by _path_residuals."""
    return dtype


@register_atom(witness_cd_path_residuals_check_array_order)
@icontract.ensure(
    lambda result, X_order: result is X_order,
    "check_array order kwarg must preserve the supplied X_order object",
)
def cd_path_residuals_check_array_order(X_order: object) -> object:
    """Return the order kwarg passed to check_array by _path_residuals."""
    return X_order


@register_atom(witness_cd_path_residuals_path_result_alphas)
@icontract.require(lambda path_result: _path_result_valid(path_result), "path_result must be a length-3 sequence")
@icontract.ensure(
    lambda result, path_result: result is path_result[0],
    "alphas unpack must preserve the first path(...) result component",
)
def cd_path_residuals_path_result_alphas(path_result: Sequence[object]) -> object:
    """Return the alpha grid unpacked from path(...) inside _path_residuals."""
    return path_result[0]


@register_atom(witness_cd_path_residuals_path_result_coefs)
@icontract.require(lambda path_result: _path_result_valid(path_result), "path_result must be a length-3 sequence")
@icontract.ensure(
    lambda result, path_result: result is path_result[1],
    "coefficient unpack must preserve the second path(...) result component",
)
def cd_path_residuals_path_result_coefs(path_result: Sequence[object]) -> object:
    """Return the coefficient path unpacked from path(...) inside _path_residuals."""
    return path_result[1]
