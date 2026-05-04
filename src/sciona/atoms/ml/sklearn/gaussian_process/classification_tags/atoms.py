"""Gaussian-process classification estimator-tag atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_gpc_binary_has_classifier_tags,
    witness_gpc_binary_target_required_tag,
    witness_gpc_classifier_estimator_type_tag,
    witness_gpc_classifier_has_classifier_tags,
    witness_gpc_classifier_target_required_tag,
)


def _bool(value: object) -> bool:
    return isinstance(value, bool)


def _estimator_type_or_none(value: object) -> bool:
    return value is None or isinstance(value, str)


@register_atom(witness_gpc_binary_target_required_tag)
@icontract.require(lambda parent_required=False: _bool(parent_required), "parent_required must be boolean")
@icontract.ensure(lambda result: _bool(result) and result is False, "binary GPC target-required tag must be False")
def gpc_binary_target_required_tag(parent_required: bool = False) -> bool:
    """Return _BinaryGaussianProcessClassifierLaplace's target-required tag override."""
    del parent_required
    return False


@register_atom(witness_gpc_binary_has_classifier_tags)
@icontract.require(
    lambda parent_has_classifier_tags=False: _bool(parent_has_classifier_tags),
    "parent_has_classifier_tags must be boolean",
)
@icontract.ensure(
    lambda result: _bool(result) and result is False,
    "binary GPC classifier-tag presence must be False",
)
def gpc_binary_has_classifier_tags(parent_has_classifier_tags: bool = False) -> bool:
    """Return whether _BinaryGaussianProcessClassifierLaplace exposes classifier-specific tags."""
    del parent_has_classifier_tags
    return False


@register_atom(witness_gpc_classifier_estimator_type_tag)
@icontract.require(
    lambda parent_estimator_type=None: _estimator_type_or_none(parent_estimator_type),
    "parent_estimator_type must be a string or None",
)
@icontract.ensure(
    lambda result: isinstance(result, str) and result == "classifier",
    "GaussianProcessClassifier estimator_type tag must be 'classifier'",
)
def gpc_classifier_estimator_type_tag(parent_estimator_type: str | None = None) -> str:
    """Return GaussianProcessClassifier's estimator-type tag override."""
    del parent_estimator_type
    return "classifier"


@register_atom(witness_gpc_classifier_target_required_tag)
@icontract.require(lambda parent_required: _bool(parent_required), "parent_required must be boolean")
@icontract.ensure(lambda result: _bool(result) and result is True, "GaussianProcessClassifier target-required tag must be True")
def gpc_classifier_target_required_tag(parent_required: bool) -> bool:
    """Return GaussianProcessClassifier's target-required tag override."""
    del parent_required
    return True


@register_atom(witness_gpc_classifier_has_classifier_tags)
@icontract.require(
    lambda parent_has_classifier_tags: _bool(parent_has_classifier_tags),
    "parent_has_classifier_tags must be boolean",
)
@icontract.ensure(
    lambda result: _bool(result) and result is True,
    "GaussianProcessClassifier classifier-tag presence must be True",
)
def gpc_classifier_has_classifier_tags(parent_has_classifier_tags: bool) -> bool:
    """Return whether GaussianProcessClassifier exposes classifier-specific tags."""
    del parent_has_classifier_tags
    return True
