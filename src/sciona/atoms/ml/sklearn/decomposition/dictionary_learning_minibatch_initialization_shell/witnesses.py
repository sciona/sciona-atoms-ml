"""Ghost witnesses for MiniBatchDictionaryLearning initialization-shell atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_dictionary_learning_minibatch_initial_dictionary(
    dict_init: AbstractArray | None,
    svd_dictionary: AbstractArray | None,
) -> AbstractArray:
    """Describe the selected initial dictionary matrix."""
    if dict_init is not None:
        return AbstractArray(shape=dict_init.shape, dtype=dict_init.dtype)
    if svd_dictionary is None:
        raise ValueError("svd_dictionary is required when dict_init is None")
    return AbstractArray(shape=svd_dictionary.shape, dtype=svd_dictionary.dtype)


def witness_dictionary_learning_minibatch_resize_dictionary(
    dictionary: AbstractArray,
    n_components: int,
) -> AbstractArray:
    """Describe the resized dictionary matrix."""
    if len(dictionary.shape) != 2:
        raise ValueError("dictionary must be rank 2")
    return AbstractArray(shape=(n_components, dictionary.shape[1]), dtype=dictionary.dtype)


def witness_dictionary_learning_minibatch_dictionary_buffer(
    dictionary: AbstractArray,
    dtype_name: str,
) -> AbstractArray:
    """Describe the writable Fortran-ordered dictionary buffer."""
    del dtype_name
    return AbstractArray(shape=dictionary.shape, dtype=dictionary.dtype)
