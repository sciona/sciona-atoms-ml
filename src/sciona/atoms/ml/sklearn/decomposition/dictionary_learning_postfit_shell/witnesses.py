"""Ghost witnesses for DictionaryLearning postfit-shell atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_dictionary_learning_fit_components(
    dictionary: AbstractArray,
) -> AbstractArray:
    """Describe DictionaryLearning.components_ postfit exposure."""
    return AbstractArray(shape=dictionary.shape, dtype=dictionary.dtype)


def witness_dictionary_learning_fit_errors(
    errors: AbstractArray,
) -> AbstractArray:
    """Describe DictionaryLearning.error_ postfit exposure."""
    return AbstractArray(shape=errors.shape, dtype=errors.dtype)


def witness_dictionary_learning_fit_n_iter(
    n_iter: int,
) -> AbstractArray:
    """Describe DictionaryLearning.n_iter_ postfit exposure."""
    del n_iter
    return AbstractArray(shape=(), dtype="int64")


def witness_dictionary_learning_fit_transform_output(
    code: AbstractArray,
) -> AbstractArray:
    """Describe DictionaryLearning.fit_transform output exposure."""
    return AbstractArray(shape=code.shape, dtype=code.dtype)
