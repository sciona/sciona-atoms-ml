"""Ghost witnesses for one-vs-rest fit bookkeeping helpers."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_one_vs_rest_binary_fit_labels(class_label: object) -> AbstractArray:
    """Describe sklearn's two-label classes argument for one binary OvR fit."""
    del class_label
    return AbstractArray(shape=(2,), dtype="object")


def witness_one_vs_rest_class_count(classes: tuple[object, ...]) -> AbstractArray:
    """Describe the scalar class count exposed by a fitted one-vs-rest classifier."""
    del classes
    return AbstractArray(shape=(), dtype="int64", min_val=1.0)


def witness_one_vs_rest_multilabel_flag(y_type: str) -> AbstractArray:
    """Describe the Boolean multilabel flag derived from a fitted label binarizer type."""
    del y_type
    return AbstractArray(shape=(), dtype="bool")


def witness_one_vs_rest_partial_fit_first_call(has_estimators: bool) -> AbstractArray:
    """Describe the Boolean first-call flag for one-vs-rest partial_fit state."""
    del has_estimators
    return AbstractArray(shape=(), dtype="bool")
