"""Covariance estimator-tag atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_covariance_estimator_type_tag,
    witness_covariance_has_classifier_tags,
    witness_covariance_target_required_tag,
    witness_elliptic_envelope_estimator_type_tag,
)


def _bool(value: object) -> bool:
    return isinstance(value, bool)


def _estimator_type_or_none(value: object) -> bool:
    return value is None or isinstance(value, str)


@register_atom(witness_covariance_target_required_tag)
@icontract.require(lambda parent_required=False: _bool(parent_required), "parent_required must be boolean")
@icontract.ensure(lambda result: _bool(result) and result is False, "covariance target-required tag must be False")
def covariance_target_required_tag(parent_required: bool = False) -> bool:
    """Return the shared target-required tag value for covariance estimators."""
    del parent_required
    return False


@register_atom(witness_covariance_estimator_type_tag)
@icontract.require(
    lambda parent_estimator_type=None: _estimator_type_or_none(parent_estimator_type),
    "parent_estimator_type must be a string or None",
)
@icontract.ensure(
    lambda result: result is None,
    "covariance estimator_type tag must be None",
)
def covariance_estimator_type_tag(parent_estimator_type: str | None = None) -> None:
    """Return the shared estimator-type tag value for covariance estimators."""
    del parent_estimator_type
    return None


@register_atom(witness_covariance_has_classifier_tags)
@icontract.require(
    lambda parent_has_classifier_tags=False: _bool(parent_has_classifier_tags),
    "parent_has_classifier_tags must be boolean",
)
@icontract.ensure(
    lambda result: _bool(result) and result is False,
    "covariance classifier-tag presence must be False",
)
def covariance_has_classifier_tags(parent_has_classifier_tags: bool = False) -> bool:
    """Return whether covariance estimators expose classifier-specific tags."""
    del parent_has_classifier_tags
    return False


@register_atom(witness_elliptic_envelope_estimator_type_tag)
@icontract.require(
    lambda parent_estimator_type=None: _estimator_type_or_none(parent_estimator_type),
    "parent_estimator_type must be a string or None",
)
@icontract.ensure(
    lambda result: isinstance(result, str) and result == "outlier_detector",
    "EllipticEnvelope estimator_type tag must be 'outlier_detector'",
)
def elliptic_envelope_estimator_type_tag(parent_estimator_type: str | None = None) -> str:
    """Return EllipticEnvelope's estimator-type tag override."""
    del parent_estimator_type
    return "outlier_detector"
