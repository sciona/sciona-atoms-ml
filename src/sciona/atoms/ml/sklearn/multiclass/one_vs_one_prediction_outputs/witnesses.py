"""Ghost witnesses for one-vs-one prediction-output helpers."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_one_vs_one_binary_labels(
    scores: AbstractArray,
    classes: AbstractArray,
    *,
    threshold: float = 0.0,
) -> AbstractArray:
    """Describe one binary OvO output label per sample."""
    del threshold
    if len(scores.shape) != 1:
        raise ValueError("scores must be 1D")
    n_samples = int(scores.shape[0])
    if n_samples < 1:
        raise ValueError("scores must be nonempty")
    if len(classes.shape) != 1 or int(classes.shape[0]) != 2:
        raise ValueError("classes must contain exactly two entries")
    return AbstractArray(shape=(n_samples,), dtype="float64")


def witness_one_vs_one_multiclass_labels(
    scores: AbstractArray,
    classes: AbstractArray,
) -> AbstractArray:
    """Describe one multiclass OvO output label per sample."""
    if len(scores.shape) != 2:
        raise ValueError("scores must be 2D")
    n_samples = int(scores.shape[0])
    n_classes = int(scores.shape[1])
    if n_samples < 1 or n_classes < 2:
        raise ValueError("scores must be nonempty with at least two classes")
    if len(classes.shape) != 1 or int(classes.shape[0]) != n_classes:
        raise ValueError("classes length must match score columns")
    return AbstractArray(shape=(n_samples,), dtype="float64")
