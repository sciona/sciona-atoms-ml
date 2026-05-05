from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError


def test_tree_prune_tree_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.tree.prune_tree_shell import (
        tree_prune_classifier_n_classes,
        tree_prune_regressor_n_classes,
        tree_prune_required,
        tree_pruned_tree_result,
    )

    assert callable(tree_prune_required)
    assert callable(tree_prune_classifier_n_classes)
    assert callable(tree_prune_regressor_n_classes)
    assert callable(tree_pruned_tree_result)


def test_tree_prune_tree_shell_matches_sklearn_runtime_setup() -> None:
    from sciona.atoms.ml.sklearn.tree.prune_tree_shell import (
        tree_prune_classifier_n_classes,
        tree_prune_regressor_n_classes,
        tree_prune_required,
        tree_pruned_tree_result,
    )

    assert tree_prune_required(0.1) is True
    assert tree_prune_required(0.0) is False
    assert np.array_equal(tree_prune_classifier_n_classes(3), np.array([3], dtype=np.intp))
    assert np.array_equal(
        tree_prune_classifier_n_classes(np.array([3, 2], dtype=np.int64)),
        np.array([3, 2], dtype=np.intp),
    )
    assert np.array_equal(tree_prune_regressor_n_classes(2), np.array([1, 1], dtype=np.intp))
    marker = object()
    assert tree_pruned_tree_result(marker) is marker


def test_tree_prune_tree_shell_contracts() -> None:
    from sciona.atoms.ml.sklearn.tree.prune_tree_shell import (
        tree_prune_classifier_n_classes,
        tree_prune_regressor_n_classes,
        tree_prune_required,
    )

    with pytest.raises(ViolationError):
        tree_prune_required(np.inf)

    with pytest.raises(ViolationError):
        tree_prune_classifier_n_classes(np.array([], dtype=np.int64))

    with pytest.raises(ViolationError):
        tree_prune_regressor_n_classes(0)
