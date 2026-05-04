"""Gaussian-process classification estimator-tag atoms."""

from .atoms import (
    gpc_binary_has_classifier_tags,
    gpc_binary_target_required_tag,
    gpc_classifier_estimator_type_tag,
    gpc_classifier_has_classifier_tags,
    gpc_classifier_target_required_tag,
)

__all__ = [
    "gpc_binary_target_required_tag",
    "gpc_binary_has_classifier_tags",
    "gpc_classifier_estimator_type_tag",
    "gpc_classifier_target_required_tag",
    "gpc_classifier_has_classifier_tags",
]
