from __future__ import annotations

import numpy as np
import pytest
from sklearn.cluster._hdbscan.hdbscan import HIERARCHY_dtype, remap_single_linkage_tree


def _tree() -> np.ndarray:
    return np.array(
        [
            (0, 1, 0.1, 2),
            (2, 3, 0.2, 3),
        ],
        dtype=HIERARCHY_dtype,
    )


def test_hdbscan_tree_remapping_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.cluster.hdbscan_tree_remapping import (
        hdbscan_outlier_linkage_rows,
        hdbscan_remapped_single_linkage_tree,
        hdbscan_remapped_tree_rows,
        hdbscan_tree_node_id,
    )

    assert callable(hdbscan_tree_node_id)
    assert callable(hdbscan_remapped_tree_rows)
    assert callable(hdbscan_outlier_linkage_rows)
    assert callable(hdbscan_remapped_single_linkage_tree)


def test_hdbscan_tree_node_id_matches_source_logic() -> None:
    from sciona.atoms.ml.sklearn.cluster.hdbscan_tree_remapping import hdbscan_tree_node_id

    internal_to_raw = {0: 2, 1: 5}
    assert hdbscan_tree_node_id(0, 2, 3, internal_to_raw) == 2
    assert hdbscan_tree_node_id(1, 2, 3, internal_to_raw) == 5
    assert hdbscan_tree_node_id(3, 2, 3, internal_to_raw) == 6


def test_hdbscan_tree_remapping_matches_sklearn_helper() -> None:
    from sciona.atoms.ml.sklearn.cluster.hdbscan_tree_remapping import (
        hdbscan_outlier_linkage_rows,
        hdbscan_remapped_single_linkage_tree,
        hdbscan_remapped_tree_rows,
    )

    tree = _tree()
    internal_to_raw = {0: 2, 1: 5}
    nonfinite = np.array([0, 4], dtype=np.int64)

    remapped_rows = hdbscan_remapped_tree_rows(tree, internal_to_raw, 2)
    assert tuple(remapped_rows["left_node"]) == (2, 4)
    assert tuple(remapped_rows["right_node"]) == (5, 5)

    outlier_rows = hdbscan_outlier_linkage_rows(6, 3, nonfinite)
    assert outlier_rows.shape == (2,)
    assert tuple(outlier_rows["left_node"]) == (0, 4)
    assert np.all(np.isinf(outlier_rows["value"]))

    expected = remap_single_linkage_tree(tree.copy(), internal_to_raw, set(int(x) for x in nonfinite))
    observed = hdbscan_remapped_single_linkage_tree(tree, internal_to_raw, nonfinite)
    assert np.array_equal(observed, expected)


def test_hdbscan_tree_remapping_contracts_reject_invalid_inputs() -> None:
    from sciona.atoms.ml.sklearn.cluster.hdbscan_tree_remapping import (
        hdbscan_outlier_linkage_rows,
        hdbscan_remapped_single_linkage_tree,
        hdbscan_tree_node_id,
    )

    with pytest.raises(Exception):
        hdbscan_tree_node_id(-1, 2, 1, {0: 1, 1: 3})

    with pytest.raises(Exception):
        hdbscan_outlier_linkage_rows(2, 0, np.array([1], dtype=np.int64))

    with pytest.raises(Exception):
        hdbscan_remapped_single_linkage_tree(_tree(), {0: 2, 1: 5}, np.array([2, 1], dtype=np.int64))
