"""Partial-dependence integer-warning shell atoms adapted from scikit-learn."""

from __future__ import annotations

from collections.abc import Sequence

import icontract

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_partial_dependence_first_integer_warning_feature,
    witness_partial_dependence_integer_warning_message,
    witness_partial_dependence_integer_warning_required,
)

FeatureLabel = int | str | bool


def _bool_scalar(value: object) -> bool:
    return isinstance(value, bool)


def _dtype_kind(value: object) -> bool:
    return isinstance(value, str) and len(value) >= 1


def _feature_label(value: object) -> bool:
    return isinstance(value, (int, str, bool))


def _feature_warning_inputs_valid(
    features: object,
    is_categorical: object,
    dtype_kinds: object,
) -> bool:
    if not (isinstance(features, tuple) and isinstance(is_categorical, tuple) and isinstance(dtype_kinds, tuple)):
        return False
    if not (len(features) == len(is_categorical) == len(dtype_kinds) and len(features) >= 1):
        return False
    return bool(
        all(_feature_label(feature) for feature in features)
        and all(_bool_scalar(flag) for flag in is_categorical)
        and all(_dtype_kind(kind) for kind in dtype_kinds)
    )


def _warning_feature_valid(
    result: object,
    features: tuple[FeatureLabel, ...],
    is_categorical: tuple[bool, ...],
    dtype_kinds: tuple[str, ...],
) -> bool:
    candidates = [
        feature
        for feature, flag, kind in zip(features, is_categorical, dtype_kinds, strict=True)
        if partial_dependence_integer_warning_required(flag, kind)
    ]
    if not candidates:
        return result is None
    return result == candidates[0]


@register_atom(witness_partial_dependence_integer_warning_required)
@icontract.require(lambda is_categorical: _bool_scalar(is_categorical), "is_categorical must be boolean")
@icontract.require(lambda dtype_kind: _dtype_kind(dtype_kind), "dtype_kind must be a nonempty dtype-kind string")
@icontract.ensure(lambda result: isinstance(result, bool), "result must be boolean")
def partial_dependence_integer_warning_required(
    is_categorical: bool,
    dtype_kind: str,
) -> bool:
    """Decide whether partial_dependence warns for one non-categorical integer-typed feature."""
    return (not is_categorical) and dtype_kind in "iu"


@register_atom(witness_partial_dependence_integer_warning_message)
@icontract.require(lambda feature: _feature_label(feature), "feature must be an int, str, or bool label")
@icontract.ensure(lambda result: isinstance(result, str) and len(result) >= 1, "result must be a nonempty warning message")
def partial_dependence_integer_warning_message(
    feature: FeatureLabel,
) -> str:
    """Format sklearn's integer-dtype FutureWarning message for partial_dependence."""
    return (
        f"The column {feature!r} contains integer data. Partial "
        "dependence plots are not supported for integer data: this "
        "can lead to implicit rounding with NumPy arrays or even errors "
        "with newer pandas versions. Please convert numerical features"
        "to floating point dtypes ahead of time to avoid problems. "
        "This will raise ValueError in scikit-learn 1.9."
    )


@register_atom(witness_partial_dependence_first_integer_warning_feature)
@icontract.require(
    lambda features, is_categorical, dtype_kinds: _feature_warning_inputs_valid(features, is_categorical, dtype_kinds),
    "features, is_categorical, and dtype_kinds must be same-length nonempty tuples",
)
@icontract.ensure(
    lambda result, features, is_categorical, dtype_kinds: _warning_feature_valid(result, features, is_categorical, dtype_kinds),
    "result must be the first feature that triggers sklearn's integer-data warning, or None",
)
def partial_dependence_first_integer_warning_feature(
    features: tuple[FeatureLabel, ...],
    is_categorical: tuple[bool, ...],
    dtype_kinds: tuple[str, ...],
) -> FeatureLabel | None:
    """Select the first feature that would trigger sklearn's integer-data warning."""
    for feature, flag, kind in zip(features, is_categorical, dtype_kinds, strict=True):
        if partial_dependence_integer_warning_required(flag, kind):
            return feature
    return None
