"""RFECV fit bookkeeping atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_rfecv_default_scoring_name,
    witness_rfecv_resolved_min_features_to_select,
    witness_rfecv_warn_min_features_too_large,
)


def _positive_feature_count(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 2


def _positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


def _nonempty_string_or_none(value: object) -> bool:
    return value is None or (isinstance(value, str) and len(value) >= 1)


def _bool_result(value: object) -> bool:
    return isinstance(value, bool)


def _resolved_count_valid(result: object, n_features: int) -> bool:
    return isinstance(result, int) and not isinstance(result, bool) and 1 <= result <= n_features


def _scoring_name_valid(result: object) -> bool:
    return isinstance(result, str) and len(result) >= 1


@register_atom(witness_rfecv_warn_min_features_too_large)
@icontract.require(lambda n_features: _positive_feature_count(n_features), "n_features must be an integer greater than or equal to 2")
@icontract.require(lambda min_features_to_select: _positive_int(min_features_to_select), "min_features_to_select must be a positive integer")
@icontract.ensure(lambda result: _bool_result(result), "warning predicate must be boolean")
def rfecv_warn_min_features_too_large(
    n_features: int,
    *,
    min_features_to_select: int,
) -> bool:
    """Return whether RFECV.fit would warn that min_features_to_select exceeds n_features."""
    return int(min_features_to_select) > int(n_features)


@register_atom(witness_rfecv_resolved_min_features_to_select)
@icontract.require(lambda n_features: _positive_feature_count(n_features), "n_features must be an integer greater than or equal to 2")
@icontract.require(lambda min_features_to_select: _positive_int(min_features_to_select), "min_features_to_select must be a positive integer")
@icontract.ensure(lambda result, n_features: _resolved_count_valid(result, n_features), "resolved feature count must lie between 1 and n_features")
def rfecv_resolved_min_features_to_select(
    n_features: int,
    *,
    min_features_to_select: int,
) -> int:
    """Resolve the min feature count RFECV passes into its inner RFE instance."""
    return min(int(min_features_to_select), int(n_features))


@register_atom(witness_rfecv_default_scoring_name)
@icontract.require(lambda estimator_is_classifier: isinstance(estimator_is_classifier, bool), "estimator_is_classifier must be boolean")
@icontract.require(lambda scoring: _nonempty_string_or_none(scoring), "scoring must be None or a nonempty string")
@icontract.ensure(lambda result: _scoring_name_valid(result), "resolved scoring name must be a nonempty string")
def rfecv_default_scoring_name(
    estimator_is_classifier: bool,
    scoring: str | None = None,
) -> str:
    """Resolve RFECV's scoring name when no explicit scoring string is supplied."""
    if scoring is None:
        return "accuracy" if estimator_is_classifier else "r2"
    return scoring
