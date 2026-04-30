"""Ghost witnesses for RFE post-fit attribute helpers."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_rfe_estimator_type(estimator_type: str) -> AbstractArray:
    """Describe RFE's exposed estimator type label."""
    del estimator_type
    return AbstractArray(shape=(), dtype="object")


def witness_rfe_classes(classes: tuple[object, ...]) -> AbstractArray:
    """Describe the fitted class-label tuple exposed through RFE.classes_."""
    del classes
    return AbstractArray(shape=(None,), dtype="object")


def witness_rfe_support_mask(support_mask: AbstractArray) -> AbstractArray:
    """Describe the fitted support mask returned by RFE._get_support_mask."""
    if len(support_mask.shape) != 1 or int(support_mask.shape[0]) < 1:
        raise ValueError("support_mask must be a nonempty 1D vector")
    return AbstractArray(shape=support_mask.shape, dtype="bool")
