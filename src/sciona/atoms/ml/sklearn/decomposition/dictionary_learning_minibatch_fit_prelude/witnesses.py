"""Ghost witnesses for MiniBatchDictionaryLearning fit-prelude atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_dictionary_learning_minibatch_training_data(
    X: AbstractArray,
    shuffle: bool,
    permutation: AbstractArray | None = None,
) -> AbstractArray:
    """Describe the minibatch training data matrix."""
    del shuffle, permutation
    return AbstractArray(shape=X.shape, dtype=X.dtype)


def witness_dictionary_learning_minibatch_old_dictionary(
    dictionary: AbstractArray,
) -> AbstractArray:
    """Describe the copied old dictionary buffer."""
    return AbstractArray(shape=dictionary.shape, dtype=dictionary.dtype)


def witness_dictionary_learning_minibatch_verbose_message(
    verbose: bool | int,
) -> AbstractArray:
    """Describe the optional verbose banner."""
    del verbose
    return AbstractArray(shape=(), dtype="object")


def witness_dictionary_learning_minibatch_monitor_state(
    max_no_improvement: int | None,
):
    """Describe the initial convergence-monitor state."""
    del max_no_improvement
    return (
        AbstractArray(shape=(), dtype="object"),
        AbstractArray(shape=(), dtype="object"),
        AbstractArray(shape=(), dtype="int64"),
    )
