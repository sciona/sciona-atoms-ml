"""Ghost witnesses for output-code book construction helpers."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_output_code_uniform_book(
    *,
    n_classes: int,
    n_estimators: int,
    seed: int,
) -> AbstractArray:
    """Describe the seeded uniform code-book matrix before sklearn discretizes it."""
    del seed
    if n_classes < 1:
        raise ValueError("n_classes must be positive")
    if n_estimators < 1:
        raise ValueError("n_estimators must be positive")
    return AbstractArray(shape=(n_classes, n_estimators), dtype="float64")


def witness_output_code_discrete_book(
    uniform_book: AbstractArray,
    *,
    has_decision_function: bool,
) -> AbstractArray:
    """Describe the discretized output-code book after sklearn thresholds the uniform draws."""
    del has_decision_function
    if len(uniform_book.shape) != 2:
        raise ValueError("uniform_book must be a matrix")
    if int(uniform_book.shape[0]) < 1 or int(uniform_book.shape[1]) < 1:
        raise ValueError("uniform_book must be nonempty")
    return AbstractArray(shape=uniform_book.shape, dtype="float64")
