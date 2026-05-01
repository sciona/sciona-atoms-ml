"""Ghost witnesses for dictionary-learning wrapper bookkeeping atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_dictionary_learning_lasso_method(method: str) -> AbstractArray:
    """Describe the lasso-prefixed method label."""
    del method
    return AbstractArray(shape=(), dtype="object")


def witness_dictionary_learning_resolved_n_components(
    X: AbstractArray,
    n_components: int | None,
) -> AbstractArray:
    """Describe the resolved dictionary component count."""
    del X, n_components
    return AbstractArray(shape=(), dtype="int64")


def witness_dict_learning_return_values(
    code: AbstractArray,
    components: AbstractArray,
    errors: AbstractArray,
    n_iter: int,
    return_n_iter: bool,
):
    """Describe dict_learning's public return packaging."""
    del n_iter, return_n_iter
    return (
        AbstractArray(shape=code.shape, dtype=code.dtype),
        AbstractArray(shape=components.shape, dtype=components.dtype),
        AbstractArray(shape=errors.shape, dtype=errors.dtype),
        AbstractArray(shape=(), dtype="int64"),
    )


def witness_dict_learning_online_return_values(
    components: AbstractArray,
    return_code: bool,
    code: AbstractArray | None = None,
):
    """Describe dict_learning_online's public return packaging."""
    del return_code, code
    return AbstractArray(shape=components.shape, dtype=components.dtype)
