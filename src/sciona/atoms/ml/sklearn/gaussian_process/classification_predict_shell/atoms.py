"""Gaussian-process classification predict shell atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_gpc_predict_dtype_name,
    witness_gpc_predict_validate_ensure_2d,
)


def _bool(value: object) -> bool:
    return isinstance(value, bool)


@register_atom(witness_gpc_predict_dtype_name)
@icontract.require(
    lambda kernel_is_none_or_requires_vector_input: _bool(kernel_is_none_or_requires_vector_input),
    "kernel_is_none_or_requires_vector_input must be boolean",
)
@icontract.ensure(
    lambda result: result in {None, "numeric"},
    "dtype mode must match sklearn's predict validation choices",
)
def gpc_predict_dtype_name(
    kernel_is_none_or_requires_vector_input: bool,
) -> str | None:
    """Resolve sklearn's predict validate_data dtype mode for Gaussian-process classification."""
    if kernel_is_none_or_requires_vector_input:
        return "numeric"
    return None


@register_atom(witness_gpc_predict_validate_ensure_2d)
@icontract.require(
    lambda kernel_is_none_or_requires_vector_input: _bool(kernel_is_none_or_requires_vector_input),
    "kernel_is_none_or_requires_vector_input must be boolean",
)
@icontract.ensure(lambda result: _bool(result), "ensure_2d mode must be boolean")
def gpc_predict_validate_ensure_2d(
    kernel_is_none_or_requires_vector_input: bool,
) -> bool:
    """Resolve sklearn's predict validate_data ensure_2d mode for Gaussian-process classification."""
    return bool(kernel_is_none_or_requires_vector_input)
