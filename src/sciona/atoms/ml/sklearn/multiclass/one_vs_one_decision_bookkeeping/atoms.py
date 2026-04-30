"""One-vs-one decision-function bookkeeping helpers adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    PairwiseIndexBlocks,
    witness_one_vs_one_decision_feature_blocks,
    witness_one_vs_one_decision_output,
)

FeatureBlocks = tuple[NDArray[np.float64], ...]
DecisionOutput = NDArray[np.float64]


def _feature_matrix_valid(values: object) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(
        array.ndim == 2
        and array.shape[0] >= 1
        and array.shape[1] >= 1
        and np.all(np.isfinite(array))
    )


def _positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


def _pairwise_indices_valid(
    pairwise_indices: object,
    X: object,
    estimator_count: int,
) -> bool:
    if pairwise_indices is None:
        return True
    if not isinstance(pairwise_indices, tuple) or len(pairwise_indices) != estimator_count:
        return False
    n_features = int(np.asarray(X, dtype=np.float64).shape[1])
    for block in pairwise_indices:
        if not isinstance(block, tuple) or len(block) < 1:
            return False
        if not all(isinstance(index, int) and not isinstance(index, bool) and 0 <= index < n_features for index in block):
            return False
    return True


def _feature_blocks_valid(
    result: object,
    X: object,
    estimator_count: int,
    pairwise_indices: object,
) -> bool:
    if not isinstance(result, tuple) or len(result) != estimator_count:
        return False
    X_values = np.asarray(X, dtype=np.float64)
    if pairwise_indices is None:
        return all(
            isinstance(block, np.ndarray)
            and block.shape == X_values.shape
            and np.allclose(block, X_values)
            for block in result
        )
    return all(
        isinstance(block, np.ndarray)
        and block.shape == (X_values.shape[0], len(indices))
        and np.allclose(block, X_values[:, indices])
        for block, indices in zip(result, pairwise_indices)
    )


def _decision_matrix_valid(values: object, n_classes: int) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(
        array.ndim == 2
        and array.shape[0] >= 1
        and array.shape[1] == n_classes
        and n_classes >= 2
        and np.all(np.isfinite(array))
    )


def _decision_output_valid(result: object, decision_scores: object, n_classes: int) -> bool:
    try:
        output = np.asarray(result, dtype=np.float64)
        scores = np.asarray(decision_scores, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    if n_classes == 2:
        return bool(output.shape == (scores.shape[0],) and np.allclose(output, scores[:, 1]))
    return bool(output.shape == scores.shape and np.allclose(output, scores))


@register_atom(witness_one_vs_one_decision_feature_blocks)
@icontract.require(lambda X: _feature_matrix_valid(X), "X must be a finite nonempty 2D dense feature matrix")
@icontract.require(lambda estimator_count: _positive_int(estimator_count), "estimator_count must be a positive integer")
@icontract.require(lambda pairwise_indices, X, estimator_count: _pairwise_indices_valid(pairwise_indices, X, estimator_count), "pairwise_indices must be None or one nonempty in-range index block per estimator")
@icontract.ensure(lambda result, X, estimator_count, pairwise_indices: _feature_blocks_valid(result, X, estimator_count, pairwise_indices), "feature blocks must match sklearn's full-X repeat or pairwise column-slice semantics")
def one_vs_one_decision_feature_blocks(
    X: NDArray[np.float64],
    *,
    estimator_count: int,
    pairwise_indices: PairwiseIndexBlocks | None = None,
) -> FeatureBlocks:
    """Build sklearn's per-estimator feature blocks inside one-vs-one decision_function."""
    X_values = np.asarray(X, dtype=np.float64)
    if pairwise_indices is None:
        return tuple(np.asarray(X_values, dtype=np.float64) for _ in range(estimator_count))
    return tuple(np.asarray(X_values[:, indices], dtype=np.float64) for indices in pairwise_indices)


@register_atom(witness_one_vs_one_decision_output)
@icontract.require(lambda decision_scores, n_classes: _decision_matrix_valid(decision_scores, n_classes), "decision_scores must be a finite sample-by-class matrix whose width matches n_classes")
@icontract.require(lambda n_classes: _positive_int(n_classes) and n_classes >= 2, "n_classes must be an integer greater than or equal to two")
@icontract.ensure(lambda result, decision_scores, n_classes: _decision_output_valid(result, decision_scores, n_classes), "output must equal the binary second-column squeeze or the multiclass decision matrix")
def one_vs_one_decision_output(
    decision_scores: NDArray[np.float64],
    *,
    n_classes: int,
) -> DecisionOutput:
    """Apply sklearn's binary squeeze to one-vs-one decision scores when n_classes == 2."""
    score_values = np.asarray(decision_scores, dtype=np.float64)
    if n_classes == 2:
        return np.asarray(score_values[:, 1], dtype=np.float64)
    return np.asarray(score_values, dtype=np.float64)
