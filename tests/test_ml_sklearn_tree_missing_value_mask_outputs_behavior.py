from __future__ import annotations

import pytest
from icontract import ViolationError


def test_tree_missing_value_mask_outputs_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.tree.missing_value_mask_outputs import (
        tree_missing_values_mask_required,
        tree_missing_values_mask_result,
        tree_missing_values_none_result,
    )

    assert callable(tree_missing_values_none_result)
    assert callable(tree_missing_values_mask_required)
    assert callable(tree_missing_values_mask_result)


def test_tree_missing_value_mask_outputs_match_sklearn_branch_logic() -> None:
    from sciona.atoms.ml.sklearn.tree.missing_value_mask_outputs import (
        tree_missing_values_mask_required,
        tree_missing_values_mask_result,
        tree_missing_values_none_result,
    )

    assert tree_missing_values_none_result(
        mask_supported=False,
        overall_sum_has_missing=True,
    ) is None
    assert tree_missing_values_none_result(
        mask_supported=True,
        overall_sum_has_missing=False,
    ) is None

    assert tree_missing_values_mask_required(
        mask_supported=True,
        overall_sum_has_missing=True,
    ) is True
    assert tree_missing_values_mask_required(
        mask_supported=True,
        overall_sum_has_missing=False,
    ) is False

    assert tree_missing_values_mask_result((False, True, False)) == (False, True, False)


def test_tree_missing_value_mask_outputs_contracts() -> None:
    from sciona.atoms.ml.sklearn.tree.missing_value_mask_outputs import (
        tree_missing_values_mask_result,
        tree_missing_values_none_result,
    )

    with pytest.raises(ViolationError):
        tree_missing_values_none_result(
            mask_supported=True,
            overall_sum_has_missing=True,
        )

    with pytest.raises(ViolationError):
        tree_missing_values_mask_result((0, 1))

