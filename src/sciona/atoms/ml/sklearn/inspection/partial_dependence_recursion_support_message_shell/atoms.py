"""Partial-dependence recursion-support message shell atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_partial_dependence_recursion_support_guard_required,
    witness_partial_dependence_supported_recursion_classes,
    witness_partial_dependence_unsupported_recursion_message,
)

VALID_METHODS = {"auto", "brute", "recursion"}


def _method_valid(value: object) -> bool:
    return isinstance(value, str) and value in VALID_METHODS


def _flag(value: object) -> bool:
    return isinstance(value, bool)


def _class_tuple_valid(value: object) -> bool:
    return (
        isinstance(value, tuple)
        and len(value) >= 1
        and all(isinstance(item, str) and len(item) >= 1 for item in value)
    )


@register_atom(witness_partial_dependence_recursion_support_guard_required)
@icontract.require(lambda method: _method_valid(method), "method must be 'auto', 'brute', or 'recursion'")
@icontract.require(lambda supports_recursion: _flag(supports_recursion), "supports_recursion must be boolean")
@icontract.ensure(lambda result: isinstance(result, bool), "result must be boolean")
def partial_dependence_recursion_support_guard_required(
    method: str,
    *,
    supports_recursion: bool,
) -> bool:
    """Decide whether sklearn raises the unsupported-recursion estimator error."""
    return method == "recursion" and not supports_recursion


@register_atom(witness_partial_dependence_supported_recursion_classes)
@icontract.require(lambda method: _method_valid(method), "method must be 'auto', 'brute', or 'recursion'")
@icontract.ensure(lambda result: _class_tuple_valid(result), "result must be a nonempty tuple of class names")
def partial_dependence_supported_recursion_classes(
    method: str,
) -> tuple[str, ...]:
    """Return sklearn's fixed supported-class tuple for recursion-mode errors."""
    del method
    return (
        "GradientBoostingClassifier",
        "GradientBoostingRegressor",
        "HistGradientBoostingClassifier",
        "HistGradientBoostingRegressor",
        "HistGradientBoostingRegressor",
        "DecisionTreeRegressor",
        "RandomForestRegressor",
    )


@register_atom(witness_partial_dependence_unsupported_recursion_message)
@icontract.require(lambda supported_classes: _class_tuple_valid(supported_classes), "supported_classes must be a nonempty tuple of class names")
@icontract.ensure(lambda result: isinstance(result, str) and len(result) >= 1, "result must be a nonempty string")
def partial_dependence_unsupported_recursion_message(
    supported_classes: tuple[str, ...],
) -> str:
    """Format sklearn's unsupported-recursion estimator ValueError message."""
    return (
        "Only the following estimators support the 'recursion' "
        "method: {}. Try using method='brute'.".format(", ".join(supported_classes))
    )
