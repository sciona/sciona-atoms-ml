"""Witnesses for sklearn multiclass one-vs-one fit bookkeeping helpers."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from sciona.ghost.abstract import AbstractArray

PairwiseIndexBlocks = tuple[tuple[int, ...], ...]


def witness_one_vs_one_fit_classes(y: AbstractArray) -> AbstractArray:
    """Describe one-vs-one fit-time unique-class extraction."""
    if len(y.shape) != 1:
        raise ValueError("y must be 1D")
    return AbstractArray(shape=(1,), dtype="float64")


def witness_one_vs_one_fit_require_multiple_classes(classes: AbstractArray) -> AbstractArray:
    """Describe the one-vs-one multiple-class validation step."""
    if len(classes.shape) != 1:
        raise ValueError("classes must be 1D")
    return classes


def witness_one_vs_one_fit_pairwise_indices(
    classes: AbstractArray,
    pairwise_indices: PairwiseIndexBlocks,
    *,
    pairwise: bool,
) -> PairwiseIndexBlocks | None:
    """Describe pairwise_indices_ selection from one-vs-one fitted worker outputs."""
    if len(classes.shape) != 1:
        raise ValueError("classes must be 1D")
    if pairwise:
        return pairwise_indices
    return None
