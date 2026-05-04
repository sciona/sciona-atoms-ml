"""Partial-dependence custom-values shell atoms adapted from scikit-learn."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import icontract

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_partial_dependence_custom_values_mapping,
    witness_partial_dependence_custom_values_subset_mapping,
    witness_partial_dependence_feature_sequence,
)

FeatureAtom = int | str | bool
FeatureItem = FeatureAtom | tuple[FeatureAtom, ...]


def _feature_atom(value: object) -> bool:
    return isinstance(value, (int, str, bool))


def _feature_item(value: object) -> bool:
    if _feature_atom(value):
        return True
    return bool(
        isinstance(value, tuple)
        and len(value) >= 1
        and all(_feature_atom(element) for element in value)
    )


def _features_input_valid(value: object) -> bool:
    if _feature_atom(value):
        return True
    return bool(
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and len(value) >= 1
        and all(_feature_item(item) for item in value)
    )


def _feature_sequence_valid(result: object) -> bool:
    return bool(
        isinstance(result, tuple)
        and len(result) >= 1
        and all(_feature_item(item) for item in result)
    )


def _custom_values_mapping_valid(value: object) -> bool:
    return value is None or isinstance(value, Mapping)


def _subset_mapping_valid(
    result: object,
    features: tuple[FeatureItem, ...],
    custom_values: Mapping[FeatureItem, object],
) -> bool:
    expected = {
        index: custom_values[feature]
        for index, feature in enumerate(features)
        if feature in custom_values
    }
    return isinstance(result, dict) and result == expected


@register_atom(witness_partial_dependence_custom_values_mapping)
@icontract.require(
    lambda custom_values=None: _custom_values_mapping_valid(custom_values),
    "custom_values must be None or a mapping",
)
@icontract.ensure(lambda result: isinstance(result, dict), "result must be a dict")
def partial_dependence_custom_values_mapping(
    custom_values: Mapping[FeatureItem, object] | None = None,
) -> dict[FeatureItem, object]:
    """Resolve sklearn's `custom_values = custom_values or {}` shell."""
    return {} if custom_values is None else dict(custom_values)


@register_atom(witness_partial_dependence_feature_sequence)
@icontract.require(lambda features: _features_input_valid(features), "features must be a scalar int/str/bool or a nonempty sequence of valid feature items")
@icontract.ensure(lambda result: _feature_sequence_valid(result), "result must be a nonempty tuple of feature items")
def partial_dependence_feature_sequence(
    features: FeatureAtom | Sequence[FeatureItem],
) -> tuple[FeatureItem, ...]:
    """Wrap sklearn scalar feature inputs and preserve sequence order otherwise."""
    if isinstance(features, (str, int)):
        return (features,)
    return tuple(features)


@register_atom(witness_partial_dependence_custom_values_subset_mapping)
@icontract.require(lambda features: _feature_sequence_valid(features), "features must be a nonempty tuple of feature items")
@icontract.require(lambda custom_values: isinstance(custom_values, Mapping), "custom_values must be a mapping")
@icontract.ensure(
    lambda result, features, custom_values: _subset_mapping_valid(result, features, custom_values),
    "result must match sklearn's indexed custom-values subset mapping",
)
def partial_dependence_custom_values_subset_mapping(
    features: tuple[FeatureItem, ...],
    custom_values: Mapping[FeatureItem, object],
) -> dict[int, object]:
    """Build the indexed custom-values mapping used for the selected feature subset."""
    return {
        index: custom_values.get(feature)
        for index, feature in enumerate(features)
        if feature in custom_values
    }
