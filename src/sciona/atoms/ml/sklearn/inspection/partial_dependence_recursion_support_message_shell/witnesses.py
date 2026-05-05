"""Ghost witnesses for partial-dependence recursion-support message shell atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_partial_dependence_recursion_support_guard_required(
    method: str,
    *,
    supports_recursion: bool,
) -> AbstractArray:
    """Describe the unsupported-recursion estimator guard predicate."""
    del method, supports_recursion
    return AbstractArray(shape=(), dtype="bool")


def witness_partial_dependence_supported_recursion_classes(
    method: str,
) -> AbstractArray:
    """Describe sklearn's fixed recursion-supported class tuple."""
    del method
    return AbstractArray(shape=(7,), dtype="str")


def witness_partial_dependence_unsupported_recursion_message(
    supported_classes: tuple[str, ...],
) -> AbstractArray:
    """Describe the unsupported-recursion estimator ValueError message."""
    if len(supported_classes) < 1:
        raise ValueError("supported_classes must be nonempty")
    return AbstractArray(shape=(), dtype="str")
