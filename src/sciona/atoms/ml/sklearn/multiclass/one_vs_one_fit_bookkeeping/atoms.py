"""One-vs-one fit bookkeeping helpers adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    PairwiseIndexBlocks,
    witness_one_vs_one_fit_classes,
    witness_one_vs_one_fit_pairwise_indices,
    witness_one_vs_one_fit_require_multiple_classes,
)


def _target_vector_valid(values: object) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 1 and array.shape[0] >= 1 and np.all(np.isfinite(array)))


def _class_vector_valid(values: object, *, min_classes: int = 1) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(
        array.ndim == 1
        and array.shape[0] >= min_classes
        and np.all(np.isfinite(array))
        and np.unique(array).shape[0] == array.shape[0]
    )


def _pairwise_flag_valid(value: object) -> bool:
    return isinstance(value, bool)


def _pairwise_index_blocks_valid(values: object, classes: object) -> bool:
    if not isinstance(values, tuple):
        return False
    class_values = np.asarray(classes, dtype=np.float64)
    expected_blocks = int(class_values.shape[0] * (class_values.shape[0] - 1) // 2)
    if len(values) != expected_blocks:
        return False
    for block in values:
        if not isinstance(block, tuple):
            return False
        if len(block) < 1:
            return False
        if not all(isinstance(index, int) and not isinstance(index, bool) and index >= 0 for index in block):
            return False
    return True


def _pairwise_indices_result_valid(result: object, pairwise: bool, pairwise_indices: object, classes: object) -> bool:
    if pairwise:
        return bool(
            _pairwise_index_blocks_valid(result, classes)
            and result == pairwise_indices
        )
    return result is None


@register_atom(witness_one_vs_one_fit_classes)
@icontract.require(lambda y: _target_vector_valid(y), "y must be a nonempty finite 1D target vector")
@icontract.ensure(lambda result: _class_vector_valid(result), "classes must be a finite unique class vector")
def one_vs_one_fit_classes(y: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return sklearn's sorted unique one-vs-one fit classes from a target vector."""
    return np.asarray(np.unique(np.asarray(y, dtype=np.float64)), dtype=np.float64)


@register_atom(witness_one_vs_one_fit_require_multiple_classes)
@icontract.require(lambda classes: _class_vector_valid(classes), "classes must be a finite unique class vector")
@icontract.ensure(lambda result: _class_vector_valid(result, min_classes=2), "validated classes must contain at least two unique class values")
def one_vs_one_fit_require_multiple_classes(
    classes: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Require sklearn's one-vs-one fit guard that rejects single-class data."""
    class_values = np.asarray(classes, dtype=np.float64)
    if class_values.shape[0] == 1:
        raise ValueError("OneVsOneClassifier can not be fit when only one class is present.")
    return np.asarray(class_values, dtype=np.float64)


@register_atom(witness_one_vs_one_fit_pairwise_indices)
@icontract.require(lambda classes: _class_vector_valid(classes, min_classes=2), "classes must be a finite unique class vector with at least two classes")
@icontract.require(lambda pairwise: _pairwise_flag_valid(pairwise), "pairwise must be boolean")
@icontract.require(lambda pairwise_indices, classes: _pairwise_index_blocks_valid(pairwise_indices, classes), "pairwise_indices must provide one nonempty nonnegative index block per class pair")
@icontract.ensure(lambda result, pairwise, pairwise_indices, classes: _pairwise_indices_result_valid(result, pairwise, pairwise_indices, classes), "pairwise indices must be preserved when pairwise mode is enabled, otherwise None")
def one_vs_one_fit_pairwise_indices(
    classes: NDArray[np.float64],
    pairwise_indices: PairwiseIndexBlocks,
    *,
    pairwise: bool,
) -> PairwiseIndexBlocks | None:
    """Select sklearn's stored pairwise_indices_ blocks from fitted one-vs-one worker outputs."""
    if pairwise:
        return pairwise_indices
    return None
