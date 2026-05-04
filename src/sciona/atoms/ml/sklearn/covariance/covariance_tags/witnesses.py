"""Ghost witnesses for covariance estimator-tag atoms."""

from __future__ import annotations


def witness_covariance_target_required_tag(parent_required: bool = False) -> bool:
    """Describe the shared target-required tag override for covariance estimators."""
    del parent_required
    return False


def witness_covariance_estimator_type_tag(parent_estimator_type: str | None = None) -> None:
    """Describe the shared estimator_type tag value for covariance estimators."""
    del parent_estimator_type
    return None


def witness_covariance_has_classifier_tags(parent_has_classifier_tags: bool = False) -> bool:
    """Describe whether covariance estimators expose classifier-specific tags."""
    del parent_has_classifier_tags
    return False


def witness_elliptic_envelope_estimator_type_tag(parent_estimator_type: str | None = None) -> str:
    """Describe EllipticEnvelope's outlier-detector estimator_type tag."""
    del parent_estimator_type
    return "outlier_detector"
