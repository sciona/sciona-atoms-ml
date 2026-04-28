"""Multioutput classifier output bookkeeping helpers adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_multioutput_classifier_probability_blocks,
    witness_multioutput_classifier_score_require_2d_targets,
    witness_multioutput_classifier_score_require_matching_output_count,
    witness_multioutput_predict_require_base_predict_method,
)

ProbabilityBlock = NDArray[np.float64]
ProbabilityBlockTuple = tuple[ProbabilityBlock, ...]


def _flag_valid(value: object) -> bool:
    return isinstance(value, bool)


def _target_matrix_like(y: object) -> bool:
    try:
        values = np.asarray(y, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(values.ndim >= 1 and values.shape[0] >= 1 and np.all(np.isfinite(values)))


def _score_target_matrix(result: object) -> bool:
    try:
        values = np.asarray(result, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(values.ndim == 2 and values.shape[0] >= 1 and values.shape[1] >= 1 and np.all(np.isfinite(values)))


def _positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


def _probability_matrix_valid(probabilities: object) -> bool:
    try:
        values = np.asarray(probabilities, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(
        values.ndim == 2
        and values.shape[0] >= 1
        and values.shape[1] >= 1
        and np.all(np.isfinite(values))
        and np.all(values >= 0.0)
        and np.allclose(np.sum(values, axis=1), 1.0)
    )


def _probability_block_sequence_valid(probability_blocks: object) -> bool:
    return bool(
        isinstance(probability_blocks, tuple)
        and len(probability_blocks) >= 1
        and all(_probability_matrix_valid(block) for block in probability_blocks)
    )


def _probability_blocks_result_valid(result: object, probability_blocks: object) -> bool:
    if not isinstance(result, tuple) or not isinstance(probability_blocks, tuple) or len(result) != len(probability_blocks):
        return False
    return all(
        _probability_matrix_valid(result_block)
        and np.array_equal(np.asarray(result_block, dtype=np.float64), np.asarray(input_block, dtype=np.float64))
        for result_block, input_block in zip(result, probability_blocks)
    )


@register_atom(witness_multioutput_predict_require_base_predict_method)
@icontract.require(lambda estimator_has_predict: _flag_valid(estimator_has_predict), "estimator_has_predict must be boolean")
@icontract.ensure(lambda result: _flag_valid(result), "result must be boolean")
def multioutput_predict_require_base_predict_method(estimator_has_predict: bool) -> bool:
    """Enforce sklearn's base-estimator predict-method requirement before multioutput predict."""
    if not estimator_has_predict:
        raise ValueError("The base estimator should implement a predict method")
    return estimator_has_predict


@register_atom(witness_multioutput_classifier_score_require_2d_targets)
@icontract.require(lambda y: _target_matrix_like(y), "y must be a finite array-like target structure with at least one sample")
@icontract.ensure(lambda result: _score_target_matrix(result), "validated score targets must be a finite nonempty 2D matrix")
def multioutput_classifier_score_require_2d_targets(y: NDArray[np.float64]) -> NDArray[np.float64]:
    """Require sklearn's 2D target shape for multioutput classifier scoring."""
    values = np.asarray(y, dtype=np.float64)
    if values.ndim == 1:
        raise ValueError(
            "y must have at least two dimensions for multi target classification but has only one"
        )
    return np.asarray(values, dtype=np.float64)


@register_atom(witness_multioutput_classifier_score_require_matching_output_count)
@icontract.require(lambda y: _score_target_matrix(y), "y must be a finite nonempty 2D target matrix")
@icontract.require(lambda n_outputs: _positive_int(n_outputs), "n_outputs must be a positive integer")
@icontract.ensure(lambda result, y: _score_target_matrix(result) and np.array_equal(np.asarray(result, dtype=np.float64), np.asarray(y, dtype=np.float64)), "validated score targets must preserve y")
def multioutput_classifier_score_require_matching_output_count(
    y: NDArray[np.float64],
    n_outputs: int,
) -> NDArray[np.float64]:
    """Require sklearn's matching output-count rule for multioutput classifier scoring."""
    values = np.asarray(y, dtype=np.float64)
    if values.shape[1] != n_outputs:
        raise ValueError(
            "The number of outputs of Y for fit {0} and score {1} should be same".format(
                n_outputs, values.shape[1]
            )
        )
    return np.asarray(values, dtype=np.float64)


@register_atom(witness_multioutput_classifier_probability_blocks)
@icontract.require(
    lambda probability_blocks: _probability_block_sequence_valid(probability_blocks),
    "probability_blocks must be a nonempty tuple of normalized nonnegative sample-by-class matrices",
)
@icontract.ensure(
    lambda result, probability_blocks: _probability_blocks_result_valid(result, probability_blocks),
    "probability blocks must preserve tuple length and each block's numeric contents",
)
def multioutput_classifier_probability_blocks(
    probability_blocks: ProbabilityBlockTuple,
) -> ProbabilityBlockTuple:
    """Canonicalize already-computed per-output predict_proba matrices for multioutput classification."""
    return tuple(np.asarray(block, dtype=np.float64) for block in probability_blocks)
