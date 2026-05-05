"""Sklearn tree missing-value support atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_tree_missing_values_allow_nan_enabled,
    witness_tree_missing_values_monotonic_constraints_absent,
    witness_tree_missing_values_supported,
    witness_tree_missing_values_x_is_sparse,
)


def _bool(value: object) -> bool:
    return isinstance(value, bool)


@register_atom(witness_tree_missing_values_x_is_sparse)
@icontract.require(lambda x_is_sparse: _bool(x_is_sparse), "x_is_sparse must be boolean")
@icontract.ensure(lambda result: _bool(result), "result must be boolean")
def tree_missing_values_x_is_sparse(x_is_sparse: bool) -> bool:
    """Return whether the tree input is sparse."""
    return x_is_sparse


@register_atom(witness_tree_missing_values_allow_nan_enabled)
@icontract.require(lambda allow_nan_tag: _bool(allow_nan_tag), "allow_nan_tag must be boolean")
@icontract.ensure(lambda result: _bool(result), "result must be boolean")
def tree_missing_values_allow_nan_enabled(allow_nan_tag: bool) -> bool:
    """Return whether the tree tag payload permits NaN input."""
    return allow_nan_tag


@register_atom(witness_tree_missing_values_monotonic_constraints_absent)
@icontract.require(
    lambda monotonic_cst_is_none: _bool(monotonic_cst_is_none),
    "monotonic_cst_is_none must be boolean",
)
@icontract.ensure(lambda result: _bool(result), "result must be boolean")
def tree_missing_values_monotonic_constraints_absent(monotonic_cst_is_none: bool) -> bool:
    """Return whether monotonic constraints are absent."""
    return monotonic_cst_is_none


@register_atom(witness_tree_missing_values_supported)
@icontract.require(lambda x_is_sparse: _bool(x_is_sparse), "x_is_sparse must be boolean")
@icontract.require(lambda allow_nan_tag: _bool(allow_nan_tag), "allow_nan_tag must be boolean")
@icontract.require(
    lambda monotonic_cst_is_none: _bool(monotonic_cst_is_none),
    "monotonic_cst_is_none must be boolean",
)
@icontract.ensure(lambda result: _bool(result), "result must be boolean")
def tree_missing_values_supported(
    *,
    x_is_sparse: bool,
    allow_nan_tag: bool,
    monotonic_cst_is_none: bool,
) -> bool:
    """Return BaseDecisionTree._support_missing_values from explicit gate inputs."""
    return (not x_is_sparse) and allow_nan_tag and monotonic_cst_is_none

