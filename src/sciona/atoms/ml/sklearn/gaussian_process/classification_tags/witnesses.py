"""Ghost witnesses for Gaussian-process classification estimator-tag atoms."""

from __future__ import annotations


def witness_gpc_binary_target_required_tag(parent_required: bool) -> bool:
    """Describe the binary Laplace estimator's target-required tag override."""
    del parent_required
    return False


def witness_gpc_binary_has_classifier_tags(parent_has_classifier_tags: bool) -> bool:
    """Describe whether the binary Laplace estimator exposes classifier-specific tags."""
    del parent_has_classifier_tags
    return False


def witness_gpc_classifier_estimator_type_tag(parent_estimator_type: str | None) -> str:
    """Describe GaussianProcessClassifier's estimator-type tag override."""
    del parent_estimator_type
    return "classifier"


def witness_gpc_classifier_target_required_tag(parent_required: bool) -> bool:
    """Describe GaussianProcessClassifier's target-required tag override."""
    del parent_required
    return True


def witness_gpc_classifier_has_classifier_tags(parent_has_classifier_tags: bool) -> bool:
    """Describe whether GaussianProcessClassifier exposes classifier-specific tags."""
    del parent_has_classifier_tags
    return True
