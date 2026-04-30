"""One-vs-rest probability bookkeeping helpers adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import witness_one_vs_rest_positive_probability_stack


ProbabilityBlock = NDArray[np.float64]
ProbabilityBlockTuple = tuple[ProbabilityBlock, ...]


def _probability_matrix_valid(values: object) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(
        array.ndim == 2
        and array.shape[0] >= 1
        and array.shape[1] >= 2
        and np.all(np.isfinite(array))
        and np.all((0.0 <= array) & (array <= 1.0))
        and np.allclose(np.sum(array, axis=1), 1.0)
    )


def _probability_block_tuple_valid(probability_blocks: object) -> bool:
    if not isinstance(probability_blocks, tuple) or len(probability_blocks) < 1:
        return False
    blocks = [np.asarray(block, dtype=np.float64) for block in probability_blocks]
    if not all(_probability_matrix_valid(block) for block in blocks):
        return False
    n_samples = blocks[0].shape[0]
    return all(block.shape[0] == n_samples for block in blocks)


def _positive_probability_stack_valid(result: object, probability_blocks: object) -> bool:
    if not isinstance(probability_blocks, tuple):
        return False
    output = np.asarray(result, dtype=np.float64)
    blocks = tuple(np.asarray(block, dtype=np.float64) for block in probability_blocks)
    return bool(
        output.shape == (len(blocks), blocks[0].shape[0])
        and np.all((0.0 <= output) & (output <= 1.0))
        and all(np.allclose(output[index], block[:, 1]) for index, block in enumerate(blocks))
    )


@register_atom(witness_one_vs_rest_positive_probability_stack)
@icontract.require(
    lambda probability_blocks: _probability_block_tuple_valid(probability_blocks),
    "probability_blocks must be a nonempty tuple of normalized sample-by-class matrices with a shared sample count",
)
@icontract.ensure(
    lambda result, probability_blocks: _positive_probability_stack_valid(result, probability_blocks),
    "positive probability stack must preserve each estimator block's positive-class column in output-by-sample layout",
)
def one_vs_rest_positive_probability_stack(
    probability_blocks: ProbabilityBlockTuple,
) -> NDArray[np.float64]:
    """Extract sklearn's per-estimator positive-class probabilities into output-by-sample layout."""
    blocks = tuple(np.asarray(block, dtype=np.float64) for block in probability_blocks)
    return np.asarray([block[:, 1] for block in blocks], dtype=np.float64)
