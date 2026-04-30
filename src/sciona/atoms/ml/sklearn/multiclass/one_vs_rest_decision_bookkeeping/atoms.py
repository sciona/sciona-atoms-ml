"""One-vs-rest decision-function bookkeeping helpers adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import witness_one_vs_rest_decision_stack


DecisionVector = NDArray[np.float64]
DecisionVectorTuple = tuple[DecisionVector, ...]


def _decision_vector_valid(values: object) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 1 and array.shape[0] >= 1 and np.all(np.isfinite(array)))


def _decision_tuple_valid(decision_blocks: object) -> bool:
    if not isinstance(decision_blocks, tuple) or len(decision_blocks) < 1:
        return False
    blocks = [np.asarray(block, dtype=np.float64) for block in decision_blocks]
    if not all(_decision_vector_valid(block) for block in blocks):
        return False
    n_samples = blocks[0].shape[0]
    return all(block.shape == (n_samples,) for block in blocks)


def _decision_stack_valid(result: object, decision_blocks: object) -> bool:
    if not isinstance(decision_blocks, tuple):
        return False
    output = np.asarray(result, dtype=np.float64)
    blocks = tuple(np.asarray(block, dtype=np.float64) for block in decision_blocks)
    return bool(
        output.shape == (len(blocks), blocks[0].shape[0])
        and np.all(np.isfinite(output))
        and all(np.allclose(output[index], block) for index, block in enumerate(blocks))
    )


@register_atom(witness_one_vs_rest_decision_stack)
@icontract.require(
    lambda decision_blocks: _decision_tuple_valid(decision_blocks),
    "decision_blocks must be a nonempty tuple of finite decision vectors with a shared sample count",
)
@icontract.ensure(
    lambda result, decision_blocks: _decision_stack_valid(result, decision_blocks),
    "decision stack must preserve each estimator's decision vector in output-by-sample layout",
)
def one_vs_rest_decision_stack(
    decision_blocks: DecisionVectorTuple,
) -> NDArray[np.float64]:
    """Stack per-estimator decision vectors into sklearn's output-by-sample OvR layout."""
    blocks = tuple(np.asarray(block, dtype=np.float64) for block in decision_blocks)
    return np.asarray(blocks, dtype=np.float64)
