"""Ghost witnesses for MiniBatchDictionaryLearning partial-fit shell atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_dictionary_learning_minibatch_partial_fit_first_call(
    has_components: bool,
) -> AbstractArray:
    """Describe the first-call branch predicate."""
    del has_components
    return AbstractArray(shape=(), dtype="bool")


def witness_dictionary_learning_minibatch_partial_fit_reset_required(
    has_components: bool,
) -> AbstractArray:
    """Describe validate_data(reset=...) selection."""
    del has_components
    return AbstractArray(shape=(), dtype="bool")


def witness_dictionary_learning_minibatch_partial_fit_initial_n_steps(
    first_call: bool,
) -> AbstractArray:
    """Describe the initial partial-fit n_steps_ value."""
    del first_call
    return AbstractArray(shape=(), dtype="int64")


def witness_dictionary_learning_minibatch_partial_fit_initial_inner_stats(
    n_components: int,
    n_features: int,
    dtype_name: str,
):
    """Describe the first-call zero inner-stat buffers."""
    del dtype_name
    return (
        AbstractArray(shape=(n_components, n_components), dtype="float64"),
        AbstractArray(shape=(n_features, n_components), dtype="float64"),
    )


def witness_dictionary_learning_minibatch_partial_fit_initial_dictionary(
    dictionary: AbstractArray,
) -> AbstractArray:
    """Describe the initialized dictionary branch."""
    return AbstractArray(shape=dictionary.shape, dtype=dictionary.dtype)


def witness_dictionary_learning_minibatch_partial_fit_existing_dictionary(
    components: AbstractArray,
) -> AbstractArray:
    """Describe the existing-components dictionary branch."""
    return AbstractArray(shape=components.shape, dtype=components.dtype)


def witness_dictionary_learning_minibatch_partial_fit_components(
    dictionary: AbstractArray,
) -> AbstractArray:
    """Describe the final components_ exposure."""
    return AbstractArray(shape=dictionary.shape, dtype=dictionary.dtype)


def witness_dictionary_learning_minibatch_partial_fit_updated_n_steps(
    n_steps: int,
) -> AbstractArray:
    """Describe the post-step n_steps_ increment."""
    del n_steps
    return AbstractArray(shape=(), dtype="int64")
