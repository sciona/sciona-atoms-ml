from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError
from sklearn.cluster._birch import _CFNode, _CFSubcluster, _split_node

from sciona.atoms.ml.sklearn.cluster.birch_split_leaf_links import (
    birch_split_leaf_link_plan,
    birch_split_next_neighbor_update_required,
    birch_split_prev_neighbor_update_required,
)


def _make_leaf_node(vectors: list[list[float]], *, branching_factor: int = 4) -> _CFNode:
    node = _CFNode(
        threshold=0.5,
        branching_factor=branching_factor,
        is_leaf=True,
        n_features=len(vectors[0]),
        dtype=np.float64,
    )
    for vector in vectors:
        node.append_subcluster(_CFSubcluster(linear_sum=np.asarray(vector, dtype=np.float64)))
    return node


def test_birch_split_leaf_link_atoms_import() -> None:
    assert callable(birch_split_prev_neighbor_update_required)
    assert callable(birch_split_next_neighbor_update_required)
    assert callable(birch_split_leaf_link_plan)


@pytest.mark.parametrize(
    ("has_prev_leaf", "has_next_leaf"),
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_birch_split_leaf_link_plan_matches_private_split_node(
    has_prev_leaf: bool,
    has_next_leaf: bool,
) -> None:
    current = _make_leaf_node([[0.0, 0.0], [3.0, 0.0], [0.0, 4.0]])
    prev_leaf = _make_leaf_node([[10.0, 10.0]]) if has_prev_leaf else None
    next_leaf = _make_leaf_node([[20.0, 20.0]]) if has_next_leaf else None

    if prev_leaf is not None:
        prev_leaf.next_leaf_ = current
        current.prev_leaf_ = prev_leaf
    if next_leaf is not None:
        current.next_leaf_ = next_leaf
        next_leaf.prev_leaf_ = current

    left_subcluster, right_subcluster = _split_node(current, threshold=0.5, branching_factor=4)
    left = left_subcluster.child_
    right = right_subcluster.child_
    plan = birch_split_leaf_link_plan(has_prev_leaf, has_next_leaf)

    assert birch_split_prev_neighbor_update_required(has_prev_leaf) is has_prev_leaf
    assert birch_split_next_neighbor_update_required(has_next_leaf) is has_next_leaf

    assert plan.left_prev_role == ("prev" if has_prev_leaf else None)
    assert plan.left_next_role == "right"
    assert plan.right_prev_role == "left"
    assert plan.right_next_role == ("next" if has_next_leaf else None)
    assert plan.prev_next_role == ("left" if has_prev_leaf else None)
    assert plan.next_prev_role == ("right" if has_next_leaf else None)

    assert left.prev_leaf_ is prev_leaf
    assert left.next_leaf_ is right
    assert right.prev_leaf_ is left
    assert right.next_leaf_ is next_leaf

    if prev_leaf is not None:
        assert prev_leaf.next_leaf_ is left
    if next_leaf is not None:
        assert next_leaf.prev_leaf_ is right


def test_birch_split_leaf_link_contracts_reject_invalid_inputs() -> None:
    with pytest.raises(ViolationError):
        birch_split_prev_neighbor_update_required(1)

    with pytest.raises(ViolationError):
        birch_split_next_neighbor_update_required("yes")

    with pytest.raises(ViolationError):
        birch_split_leaf_link_plan(False, None)  # type: ignore[arg-type]
