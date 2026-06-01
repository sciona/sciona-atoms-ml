"""Ghost witnesses for sklearn RANSAC predict/score callback atoms."""

from __future__ import annotations


def witness_ransac_public_validation_kwargs(method_name: str) -> str:
    """Describe fixed validate_data kwargs for public predict/score methods."""
    return method_name


def witness_ransac_public_nonrouting_params(method_name: str) -> str:
    """Describe non-routing public predict/score parameter fallback."""
    return method_name


def witness_ransac_public_predict_callback_payload(estimator: object, X: object, predict_params: object) -> object:
    """Describe public estimator_.predict callback payload."""
    return (estimator, X, predict_params)


def witness_ransac_public_score_callback_payload(estimator: object, X: object, y: object, score_params: object) -> object:
    """Describe public estimator_.score callback payload."""
    return (estimator, X, y, score_params)
