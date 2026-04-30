"""Ghost witnesses for RFECV fit bookkeeping atoms."""

from __future__ import annotations


def witness_rfecv_warn_min_features_too_large(
    n_features: int,
    *,
    min_features_to_select: int,
) -> bool:
    """Describe RFECV's warning predicate for oversized min_features_to_select."""
    if n_features < 2:
        raise ValueError("n_features must be at least 2")
    if min_features_to_select < 1:
        raise ValueError("min_features_to_select must be positive")
    return False


def witness_rfecv_resolved_min_features_to_select(
    n_features: int,
    *,
    min_features_to_select: int,
) -> int:
    """Describe RFECV's resolved min feature count passed to the inner RFE."""
    if n_features < 2:
        raise ValueError("n_features must be at least 2")
    if min_features_to_select < 1:
        raise ValueError("min_features_to_select must be positive")
    return 1


def witness_rfecv_default_scoring_name(
    estimator_is_classifier: bool,
    scoring: str | None = None,
) -> str:
    """Describe RFECV's resolved scoring name when scoring is None or explicit."""
    if not isinstance(estimator_is_classifier, bool):
        raise ValueError("estimator_is_classifier must be boolean")
    if scoring is not None and (not isinstance(scoring, str) or len(scoring) < 1):
        raise ValueError("scoring must be None or a nonempty string")
    return "accuracy" if scoring is None and estimator_is_classifier else "r2" if scoring is None else scoring
