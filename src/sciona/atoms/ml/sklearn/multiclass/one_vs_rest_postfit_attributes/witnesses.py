"""Witnesses for sklearn multiclass one-vs-rest post-fit attributes."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_one_vs_rest_fit_classes(classes: AbstractArray) -> AbstractArray:
    """Describe one-vs-rest fitted classes_ pass-through."""
    if len(classes.shape) != 1:
        raise ValueError("classes must be 1D")
    return classes


def witness_one_vs_rest_fit_n_features_in(n_features_in: int) -> int:
    """Describe one-vs-rest fitted n_features_in_ pass-through."""
    return int(n_features_in)


def witness_one_vs_rest_fit_feature_names_in(feature_names_in: tuple[str, ...]) -> tuple[str, ...]:
    """Describe one-vs-rest fitted feature_names_in_ pass-through."""
    return tuple(feature_names_in)
