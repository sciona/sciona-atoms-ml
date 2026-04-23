"""Ghost witnesses for dictionary-learning update helper atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def _check_matrix(values: AbstractArray, name: str) -> tuple[int, int]:
    if len(values.shape) != 2:
        raise ValueError(f"{name} must be 2D")
    rows, cols = int(values.shape[0]), int(values.shape[1])
    if rows < 1 or cols < 1:
        raise ValueError(f"{name} must be nonempty")
    return rows, cols


def witness_dictionary_learning_sufficient_statistics(
    Y: AbstractArray,
    code: AbstractArray,
) -> tuple[AbstractArray, AbstractArray]:
    """Describe dictionary-learning sufficient statistics."""
    n_samples, n_features = _check_matrix(Y, "Y")
    code_samples, n_components = _check_matrix(code, "code")
    if code_samples != n_samples:
        raise ValueError("Y and code must have matching sample count")
    return (
        AbstractArray(shape=(n_components, n_components), dtype="float64"),
        AbstractArray(shape=(n_features, n_components), dtype="float64"),
    )


def witness_dictionary_learning_active_update(
    dictionary: AbstractArray,
    A: AbstractArray,
    B: AbstractArray,
    *,
    positive: bool = False,
) -> AbstractArray:
    """Describe active dictionary atom updates."""
    del positive
    n_components, n_features = _check_matrix(dictionary, "dictionary")
    a_rows, a_cols = _check_matrix(A, "A")
    b_rows, b_cols = _check_matrix(B, "B")
    if (a_rows, a_cols) != (n_components, n_components):
        raise ValueError("A must be square over components")
    if (b_rows, b_cols) != (n_features, n_components):
        raise ValueError("B must be features by components")
    return AbstractArray(shape=(n_components, n_features), dtype="float64")
