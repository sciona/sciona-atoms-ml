"""Sklearn tree missing-value-mask prelude atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_tree_missing_values_common_kwargs,
    witness_tree_missing_values_estimator_name,
    witness_tree_missing_values_overall_sum_has_missing,
    witness_tree_missing_values_overall_sum_requires_elementwise_check,
)


def _optional_name(value: object) -> bool:
    return value is None or (isinstance(value, str) and len(value) >= 1)


def _nonempty_name(value: object) -> bool:
    return isinstance(value, str) and len(value) >= 1


def _finite_or_special_scalar(value: object) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)


@register_atom(witness_tree_missing_values_estimator_name)
@icontract.require(lambda estimator_name=None: _optional_name(estimator_name), "estimator_name must be None or a nonempty string")
@icontract.require(lambda class_name: _nonempty_name(class_name), "class_name must be a nonempty string")
@icontract.ensure(lambda result: _nonempty_name(result), "resolved estimator_name must be a nonempty string")
def tree_missing_values_estimator_name(
    class_name: str,
    estimator_name: str | None = None,
) -> str:
    """Resolve estimator_name defaulting in _compute_missing_values_in_feature_mask."""
    return estimator_name or class_name


@register_atom(witness_tree_missing_values_common_kwargs)
@icontract.require(lambda estimator_name: _nonempty_name(estimator_name), "estimator_name must be a nonempty string")
@icontract.ensure(
    lambda result, estimator_name: isinstance(result, dict)
    and result == {"estimator_name": estimator_name, "input_name": "X"},
    "common_kwargs must match sklearn's estimator_name and input_name payload",
)
def tree_missing_values_common_kwargs(estimator_name: str) -> dict[str, str]:
    """Construct common_kwargs for sklearn's finite-value checks."""
    return {"estimator_name": estimator_name, "input_name": "X"}


@register_atom(witness_tree_missing_values_overall_sum_requires_elementwise_check)
@icontract.require(lambda overall_sum: _finite_or_special_scalar(overall_sum), "overall_sum must be a numeric scalar")
@icontract.ensure(lambda result: isinstance(result, bool), "result must be boolean")
def tree_missing_values_overall_sum_requires_elementwise_check(overall_sum: float) -> bool:
    """Return whether sklearn escalates to elementwise finite checking from overall_sum."""
    return not bool(np.isfinite(overall_sum))


@register_atom(witness_tree_missing_values_overall_sum_has_missing)
@icontract.require(lambda overall_sum: _finite_or_special_scalar(overall_sum), "overall_sum must be a numeric scalar")
@icontract.ensure(lambda result: isinstance(result, bool), "result must be boolean")
def tree_missing_values_overall_sum_has_missing(overall_sum: float) -> bool:
    """Return whether sklearn's overall_sum indicates missing values via NaN."""
    return bool(np.isnan(overall_sum))

