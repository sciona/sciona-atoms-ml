"""Sklearn tree fitted-property atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_tree_get_depth_result,
    witness_tree_get_n_leaves_result,
)


def _nonnegative_int(value: object) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool) and int(value) >= 0


def _positive_int(value: object) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool) and int(value) >= 1


@register_atom(witness_tree_get_depth_result)
@icontract.require(lambda tree_max_depth: _nonnegative_int(tree_max_depth), "tree_max_depth must be a nonnegative integer")
@icontract.ensure(
    lambda result, tree_max_depth: _nonnegative_int(result) and result == tree_max_depth,
    "get_depth result must preserve the fitted tree max_depth",
)
def tree_get_depth_result(tree_max_depth: int) -> int:
    """Return BaseDecisionTree.get_depth from the fitted tree max_depth value."""
    return tree_max_depth


@register_atom(witness_tree_get_n_leaves_result)
@icontract.require(lambda tree_n_leaves: _positive_int(tree_n_leaves), "tree_n_leaves must be a positive integer")
@icontract.ensure(
    lambda result, tree_n_leaves: _positive_int(result) and result == tree_n_leaves,
    "get_n_leaves result must preserve the fitted tree n_leaves value",
)
def tree_get_n_leaves_result(tree_n_leaves: int) -> int:
    """Return BaseDecisionTree.get_n_leaves from the fitted tree n_leaves value."""
    return tree_n_leaves
