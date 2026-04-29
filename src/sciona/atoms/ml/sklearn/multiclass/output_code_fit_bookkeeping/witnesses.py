"""Witnesses for sklearn multiclass output-code fit bookkeeping helpers."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_output_code_fit_require_nonempty_classes(classes: AbstractArray) -> AbstractArray:
    """Describe output-code fit-time class-vector validation."""
    if len(classes.shape) != 1:
        raise ValueError("classes must be 1D")
    return classes


def witness_output_code_fit_estimator_count(n_classes: int, code_size: float) -> int:
    """Describe output-code fit-time estimator-count resolution."""
    return int(n_classes * code_size)


def witness_output_code_fit_n_features_in(n_features_in: int) -> int:
    """Describe output-code fit-time n_features_in_ pass-through."""
    return int(n_features_in)


def witness_output_code_fit_feature_names_in(feature_names_in: tuple[str, ...]) -> tuple[str, ...]:
    """Describe output-code fit-time feature_names_in_ pass-through."""
    return tuple(feature_names_in)
