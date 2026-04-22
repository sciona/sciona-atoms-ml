"""Ghost witnesses for sklearn permutation-importance helpers."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def _check_matrix(values: AbstractArray, name: str) -> tuple[int, int]:
    if len(values.shape) != 2:
        raise ValueError(f"{name} must be 2D")
    rows, cols = int(values.shape[0]), int(values.shape[1])
    if rows < 1 or cols < 1:
        raise ValueError(f"{name} must be nonempty")
    return rows, cols


def witness_permutation_importance_values(
    baseline_score: float,
    permuted_scores: AbstractArray,
) -> AbstractArray:
    """Describe raw feature importance values from baseline and permuted scores."""
    del baseline_score
    n_features, n_repeats = _check_matrix(permuted_scores, "permuted_scores")
    return AbstractArray(shape=(n_features, n_repeats), dtype="float64")


def witness_permutation_importance_mean(importances: AbstractArray) -> AbstractArray:
    """Describe mean feature importance over repeated permutations."""
    n_features, _ = _check_matrix(importances, "importances")
    return AbstractArray(shape=(n_features,), dtype="float64")


def witness_permutation_importance_std(importances: AbstractArray) -> AbstractArray:
    """Describe spread of feature importance over repeated permutations."""
    n_features, _ = _check_matrix(importances, "importances")
    return AbstractArray(shape=(n_features,), dtype="float64")


def witness_permutation_importance_summary(
    baseline_score: float,
    permuted_scores: AbstractArray,
) -> tuple[AbstractArray, AbstractArray, AbstractArray]:
    """Describe all published permutation-importance summary arrays."""
    del baseline_score
    n_features, n_repeats = _check_matrix(permuted_scores, "permuted_scores")
    return (
        AbstractArray(shape=(n_features,), dtype="float64"),
        AbstractArray(shape=(n_features,), dtype="float64"),
        AbstractArray(shape=(n_features, n_repeats), dtype="float64"),
    )
