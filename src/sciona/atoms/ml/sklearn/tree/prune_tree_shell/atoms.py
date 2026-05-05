"""Sklearn tree pruning atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_tree_prune_classifier_n_classes,
    witness_tree_prune_regressor_n_classes,
    witness_tree_prune_required,
    witness_tree_pruned_tree_result,
)


def _nonnegative_int_vector(values: object) -> bool:
    array = np.asarray(values)
    return bool(
        array.ndim == 1
        and array.shape[0] >= 1
        and np.issubdtype(array.dtype, np.integer)
        and np.all(array >= 0)
    )


def _positive_int(value: object) -> bool:
    return isinstance(value, (int, np.integer)) and int(value) >= 1


@register_atom(witness_tree_prune_required)
@icontract.require(
    lambda ccp_alpha: np.isfinite(float(ccp_alpha)),
    "ccp_alpha must be finite",
)
@icontract.ensure(
    lambda result, ccp_alpha: isinstance(result, bool) and result == (float(ccp_alpha) != 0.0),
    "prune_required must be true exactly when ccp_alpha is nonzero",
)
def tree_prune_required(ccp_alpha: float) -> bool:
    """Return whether _prune_tree should proceed past the zero-alpha early return."""
    return float(ccp_alpha) != 0.0


@register_atom(witness_tree_prune_classifier_n_classes)
@icontract.require(
    lambda n_classes: _positive_int(n_classes) or _nonnegative_int_vector(n_classes),
    "n_classes must be a positive integer or nonempty nonnegative integer vector",
)
@icontract.ensure(
    lambda result: _nonnegative_int_vector(result),
    "classifier prune n_classes must be a one-dimensional integer vector",
)
def tree_prune_classifier_n_classes(
    n_classes: int | NDArray[np.integer],
) -> NDArray[np.intp]:
    """Return the classifier n_classes vector used to construct a pruned tree."""
    return np.atleast_1d(np.asarray(n_classes, dtype=np.intp))


@register_atom(witness_tree_prune_regressor_n_classes)
@icontract.require(
    lambda n_outputs: _positive_int(n_outputs),
    "n_outputs must be a positive integer",
)
@icontract.ensure(
    lambda result, n_outputs: _nonnegative_int_vector(result)
    and np.asarray(result, dtype=np.intp).shape == (int(n_outputs),)
    and np.array_equal(np.asarray(result, dtype=np.intp), np.ones(int(n_outputs), dtype=np.intp)),
    "regressor prune n_classes must be an all-ones intp vector sized to n_outputs",
)
def tree_prune_regressor_n_classes(n_outputs: int) -> NDArray[np.intp]:
    """Return the regressor class-count vector used to construct a pruned tree."""
    return np.array([1] * int(n_outputs), dtype=np.intp)


@register_atom(witness_tree_pruned_tree_result)
@icontract.ensure(
    lambda result, pruned_tree: result is pruned_tree,
    "pruned tree result must preserve the supplied tree object",
)
def tree_pruned_tree_result(pruned_tree: object) -> object:
    """Return the pruned tree object assigned back onto the estimator."""
    return pruned_tree
