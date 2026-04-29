"""Ghost witnesses for MLP early-stopping state helper atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def _check_matrix(values: AbstractArray, name: str) -> tuple[int, ...]:
    if len(values.shape) not in {1, 2}:
        raise ValueError(f"{name} must be 1D or 2D")
    if int(values.shape[0]) < 1:
        raise ValueError(f"{name} must be nonempty")
    return tuple(int(dim) for dim in values.shape)


def _check_parameter_sequence(values: tuple[AbstractArray, ...], name: str) -> None:
    if len(values) < 1:
        raise ValueError(f"{name} must be nonempty")
    for array in values:
        if len(array.shape) not in {1, 2}:
            raise ValueError(f"{name} entries must be 1D or 2D")


def witness_mlp_stochastic_validation_targets(
    y_val: AbstractArray,
    *,
    is_classifier: bool,
    label_binarizer_state: object | None = None,
) -> AbstractArray:
    """Describe validation targets after the optional classifier decode step."""
    del label_binarizer_state
    shape = _check_matrix(y_val, "y_val")
    if is_classifier:
        return AbstractArray(shape=(shape[0],), dtype="object")
    return AbstractArray(shape=shape, dtype="float64")


def witness_mlp_validation_scores_append(
    validation_scores: tuple[float, ...],
    val_score: float,
) -> tuple[float, ...]:
    """Describe appending one validation score to the score history."""
    del val_score
    return tuple(validation_scores) + (0.0,)


def witness_mlp_monitor_best_state(
    last_valid_score: float,
    best_validation_score: float,
    best_coefs: tuple[AbstractArray, ...],
    best_intercepts: tuple[AbstractArray, ...],
    coefs: tuple[AbstractArray, ...],
    intercepts: tuple[AbstractArray, ...],
) -> tuple[float, tuple[AbstractArray, ...], tuple[AbstractArray, ...]]:
    """Describe the cached best-score and best-parameter update state."""
    del last_valid_score, best_validation_score
    _check_parameter_sequence(best_coefs, "best_coefs")
    _check_parameter_sequence(best_intercepts, "best_intercepts")
    _check_parameter_sequence(coefs, "coefs")
    _check_parameter_sequence(intercepts, "intercepts")
    if not (
        len(best_coefs) == len(best_intercepts) == len(coefs) == len(intercepts)
    ):
        raise ValueError("parameter sequences must have matching layer counts")
    return 0.0, coefs, intercepts


def witness_mlp_restore_best_parameters(
    best_coefs: tuple[AbstractArray, ...],
    best_intercepts: tuple[AbstractArray, ...],
) -> tuple[tuple[AbstractArray, ...], tuple[AbstractArray, ...]]:
    """Describe restoring the cached best parameter tuples."""
    _check_parameter_sequence(best_coefs, "best_coefs")
    _check_parameter_sequence(best_intercepts, "best_intercepts")
    if len(best_coefs) != len(best_intercepts):
        raise ValueError("best_coefs and best_intercepts must have matching layer counts")
    return best_coefs, best_intercepts
