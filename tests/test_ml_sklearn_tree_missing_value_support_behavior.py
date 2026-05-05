from __future__ import annotations

import pytest
from icontract import ViolationError
from sklearn.tree import DecisionTreeClassifier


def test_tree_missing_value_support_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.tree.missing_value_support import (
        tree_missing_values_allow_nan_enabled,
        tree_missing_values_monotonic_constraints_absent,
        tree_missing_values_supported,
        tree_missing_values_x_is_sparse,
    )

    assert callable(tree_missing_values_x_is_sparse)
    assert callable(tree_missing_values_allow_nan_enabled)
    assert callable(tree_missing_values_monotonic_constraints_absent)
    assert callable(tree_missing_values_supported)


def test_tree_missing_value_support_matches_sklearn_gate_logic() -> None:
    from sciona.atoms.ml.sklearn.tree.missing_value_support import (
        tree_missing_values_allow_nan_enabled,
        tree_missing_values_monotonic_constraints_absent,
        tree_missing_values_supported,
        tree_missing_values_x_is_sparse,
    )

    est = DecisionTreeClassifier(splitter="best", criterion="gini")
    allow_nan = est.__sklearn_tags__().input_tags.allow_nan

    assert tree_missing_values_x_is_sparse(False) is False
    assert tree_missing_values_allow_nan_enabled(allow_nan) is True
    assert tree_missing_values_monotonic_constraints_absent(True) is True
    assert (
        tree_missing_values_supported(
            x_is_sparse=False,
            allow_nan_tag=allow_nan,
            monotonic_cst_is_none=True,
        )
        is True
    )


def test_tree_missing_value_support_false_cases() -> None:
    from sciona.atoms.ml.sklearn.tree.missing_value_support import (
        tree_missing_values_supported,
    )

    assert tree_missing_values_supported(
        x_is_sparse=True,
        allow_nan_tag=True,
        monotonic_cst_is_none=True,
    ) is False
    assert tree_missing_values_supported(
        x_is_sparse=False,
        allow_nan_tag=False,
        monotonic_cst_is_none=True,
    ) is False
    assert tree_missing_values_supported(
        x_is_sparse=False,
        allow_nan_tag=True,
        monotonic_cst_is_none=False,
    ) is False


def test_tree_missing_value_support_contracts() -> None:
    from sciona.atoms.ml.sklearn.tree.missing_value_support import (
        tree_missing_values_supported,
    )

    with pytest.raises(ViolationError):
        tree_missing_values_supported(
            x_is_sparse=0,
            allow_nan_tag=True,
            monotonic_cst_is_none=True,
        )

