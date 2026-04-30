"""Witnesses for sklearn multiclass one-vs-one decision bookkeeping helpers."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray

PairwiseIndexBlocks = tuple[tuple[int, ...], ...]


def witness_one_vs_one_decision_feature_blocks(
    X: AbstractArray,
    *,
    estimator_count: int,
    pairwise_indices: PairwiseIndexBlocks | None = None,
) -> tuple[AbstractArray, ...]:
    """Describe feature-block selection inside OneVsOneClassifier.decision_function."""
    if len(X.shape) != 2:
        raise ValueError("X must be two-dimensional")
    n_samples = int(X.shape[0])
    n_features = int(X.shape[1])
    if estimator_count < 1:
        raise ValueError("estimator_count must be positive")
    if pairwise_indices is None:
        return tuple(
            AbstractArray(shape=(n_samples, n_features), dtype="float64")
            for _ in range(estimator_count)
        )
    if len(pairwise_indices) != estimator_count:
        raise ValueError("pairwise_indices must match estimator_count")
    return tuple(
        AbstractArray(shape=(n_samples, len(block)), dtype="float64")
        for block in pairwise_indices
    )


def witness_one_vs_one_decision_output(
    decision_scores: AbstractArray,
    *,
    n_classes: int,
) -> AbstractArray:
    """Describe binary squeeze versus multiclass passthrough for decision outputs."""
    if len(decision_scores.shape) != 2:
        raise ValueError("decision_scores must be two-dimensional")
    if int(decision_scores.shape[1]) != n_classes:
        raise ValueError("decision_scores width must match n_classes")
    if n_classes == 2:
        return AbstractArray(shape=(int(decision_scores.shape[0]),), dtype="float64")
    return AbstractArray(shape=decision_scores.shape, dtype="float64")
