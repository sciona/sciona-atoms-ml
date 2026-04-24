"""Ghost witnesses for dictionary-learning loop helper atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def _check_matrix(values: AbstractArray, name: str) -> tuple[int, int]:
    if len(values.shape) != 2:
        raise ValueError(f"{name} must be 2D")
    rows, cols = int(values.shape[0]), int(values.shape[1])
    if rows < 1 or cols < 1:
        raise ValueError(f"{name} must be nonempty")
    return rows, cols


def witness_dictionary_learning_svd_initialize(X: AbstractArray) -> tuple[AbstractArray, AbstractArray]:
    """Describe the initial code and dictionary factors built from an SVD."""
    rows, cols = _check_matrix(X, "X")
    rank = min(rows, cols)
    return (
        AbstractArray(shape=(rows, rank), dtype="float64"),
        AbstractArray(shape=(rank, cols), dtype="float64"),
    )


def witness_dictionary_learning_resize_factors(
    code: AbstractArray,
    dictionary: AbstractArray,
    *,
    n_components: int,
) -> tuple[AbstractArray, AbstractArray]:
    """Describe cropping or zero-padding the initial factors to n_components."""
    rows, rank_code = _check_matrix(code, "code")
    rank_dict, cols = _check_matrix(dictionary, "dictionary")
    if rank_code != rank_dict:
        raise ValueError("code width must match dictionary height")
    if n_components < 1:
        raise ValueError("n_components must be positive")
    return (
        AbstractArray(shape=(rows, n_components), dtype="float64"),
        AbstractArray(shape=(n_components, cols), dtype="float64"),
    )


def witness_dictionary_learning_cost(
    X: AbstractArray,
    code: AbstractArray,
    dictionary: AbstractArray,
    *,
    alpha: float,
) -> float:
    """Describe the scalar dictionary-learning objective value."""
    rows_x, cols_x = _check_matrix(X, "X")
    rows_c, cols_c = _check_matrix(code, "code")
    rows_d, cols_d = _check_matrix(dictionary, "dictionary")
    if rows_x != rows_c or cols_x != cols_d or cols_c != rows_d:
        raise ValueError("X, code, and dictionary must be multiplicatively compatible")
    del alpha
    return 0.0


def witness_dictionary_learning_converged(
    previous_cost: float,
    current_cost: float,
    *,
    tol: float,
) -> bool:
    """Describe the dictionary-learning cost-delta stopping predicate."""
    del previous_cost, current_cost, tol
    return False


def witness_dictionary_learning_callback_due(iteration: int) -> bool:
    """Describe the callback cadence in the dictionary-learning loop."""
    if iteration < 0:
        raise ValueError("iteration must be nonnegative")
    return False
