"""Witnesses for sklearn multiclass one-vs-rest partial-fit bookkeeping."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_one_vs_rest_partial_fit_estimator_count(n_classes: int) -> int:
    """Describe first-call estimator allocation count in OvR partial_fit."""
    if n_classes < 1:
        raise ValueError("n_classes must be positive")
    return int(n_classes)


def witness_one_vs_rest_partial_fit_label_binarizer_classes(
    classes: AbstractArray,
) -> AbstractArray:
    """Describe the class vector used to fit OvR's sparse label binarizer."""
    if len(classes.shape) != 1 or int(classes.shape[0]) < 1:
        raise ValueError("classes must be a nonempty 1D array")
    return AbstractArray(shape=(int(classes.shape[0]),), dtype="float64")


def witness_one_vs_rest_partial_fit_n_features_in(n_features_in: int) -> int:
    """Describe the fitted n_features_in_ value copied from the first estimator."""
    if n_features_in < 1:
        raise ValueError("n_features_in must be positive")
    return int(n_features_in)
