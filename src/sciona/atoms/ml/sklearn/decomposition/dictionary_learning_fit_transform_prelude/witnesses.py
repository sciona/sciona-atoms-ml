"""Witnesses for DictionaryLearning fit-transform prelude helpers."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_dictionary_learning_fit_transform_require_positive_coding_supported(
    fit_algorithm: str,
    positive_code: bool,
) -> AbstractArray:
    """Describe the positive-coding support guard."""
    del fit_algorithm, positive_code
    return AbstractArray(shape=(), dtype="bool")


def witness_dictionary_learning_fit_transform_method(
    fit_algorithm: str,
) -> AbstractArray:
    """Describe lasso-method label resolution."""
    del fit_algorithm
    return AbstractArray(shape=(), dtype="object")


def witness_dictionary_learning_fit_transform_validated_data(
    X: AbstractArray,
) -> AbstractArray:
    """Describe the validated training matrix."""
    return AbstractArray(shape=X.shape, dtype=X.dtype)


def witness_dictionary_learning_fit_transform_n_components(
    X: AbstractArray,
    n_components: int | None,
) -> AbstractArray:
    """Describe estimator-side n_components resolution."""
    del X, n_components
    return AbstractArray(shape=(), dtype="int64")
