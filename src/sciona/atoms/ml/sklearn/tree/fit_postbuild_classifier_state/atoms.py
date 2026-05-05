"""Sklearn tree post-build classifier-state atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_tree_fit_single_output_classes,
    witness_tree_fit_single_output_classifier_branch,
    witness_tree_fit_single_output_n_classes,
)


def _positive_int(value: object) -> bool:
    return isinstance(value, (int, np.integer)) and int(value) >= 1


def _nonempty_1d(values: object) -> bool:
    array = np.asarray(values)
    return bool(array.ndim == 1 and array.shape[0] >= 1)


def _nonempty_2d(values: object) -> bool:
    array = np.asarray(values, dtype=object)
    return bool(array.ndim == 2 and array.shape[0] >= 1 and array.shape[1] >= 1)


@register_atom(witness_tree_fit_single_output_classifier_branch)
@icontract.require(lambda n_outputs: _positive_int(n_outputs), "n_outputs must be a positive integer")
@icontract.require(lambda is_classifier: isinstance(is_classifier, bool), "is_classifier must be boolean")
@icontract.ensure(
    lambda result, n_outputs, is_classifier: isinstance(result, bool)
    and result == (int(n_outputs) == 1 and is_classifier),
    "single-output classifier branch must match n_outputs == 1 and is_classifier",
)
def tree_fit_single_output_classifier_branch(n_outputs: int, is_classifier: bool) -> bool:
    """Return whether BaseDecisionTree._fit should collapse classifier state."""
    return int(n_outputs) == 1 and is_classifier


@register_atom(witness_tree_fit_single_output_n_classes)
@icontract.require(
    lambda n_classes: _nonempty_1d(n_classes) and np.issubdtype(np.asarray(n_classes).dtype, np.integer),
    "n_classes must be a nonempty one-dimensional integer vector",
)
@icontract.ensure(
    lambda result, n_classes: _positive_int(result) and int(result) == int(np.asarray(n_classes)[0]),
    "single-output n_classes must equal the first class-count entry",
)
def tree_fit_single_output_n_classes(n_classes: NDArray[np.integer]) -> int:
    """Return the collapsed scalar n_classes_ value for single-output classification."""
    return int(np.asarray(n_classes, dtype=np.intp)[0])


@register_atom(witness_tree_fit_single_output_classes)
@icontract.require(
    lambda classes: _nonempty_2d(classes),
    "classes must be a nonempty two-dimensional class block array",
)
@icontract.ensure(
    lambda result, classes: _nonempty_1d(result)
    and np.array_equal(np.asarray(result, dtype=object), np.asarray(classes, dtype=object)[0]),
    "single-output classes must equal the first class block",
)
def tree_fit_single_output_classes(classes: NDArray[np.object_]) -> NDArray[np.object_]:
    """Return the collapsed classes_ vector for single-output classification."""
    return np.asarray(np.asarray(classes, dtype=object)[0], dtype=object)
