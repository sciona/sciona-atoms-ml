"""Ghost witnesses for MiniBatchDictionaryLearning postfit-shell atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_dictionary_learning_minibatch_postfit_components(
    dictionary: AbstractArray,
) -> AbstractArray:
    """Describe MiniBatchDictionaryLearning.components_ exposure."""
    return AbstractArray(shape=dictionary.shape, dtype=dictionary.dtype)


def witness_dictionary_learning_minibatch_postfit_n_steps(
    n_steps: int,
) -> AbstractArray:
    """Describe MiniBatchDictionaryLearning.n_steps_ exposure."""
    del n_steps
    return AbstractArray(shape=(), dtype="int64")


def witness_dictionary_learning_minibatch_postfit_n_iter(
    n_iter: float,
) -> AbstractArray:
    """Describe MiniBatchDictionaryLearning.n_iter_ exposure."""
    del n_iter
    return AbstractArray(shape=(), dtype="float64")


def witness_dictionary_learning_minibatch_fit_return_self(
    estimator_token: str,
) -> AbstractArray:
    """Describe the fit-return self passthrough."""
    del estimator_token
    return AbstractArray(shape=(), dtype="object")
