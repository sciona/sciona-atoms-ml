from __future__ import annotations

import math

import pytest
from icontract import ViolationError


def test_tree_missing_value_mask_prelude_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.tree.missing_value_mask_prelude import (
        tree_missing_values_common_kwargs,
        tree_missing_values_estimator_name,
        tree_missing_values_overall_sum_has_missing,
        tree_missing_values_overall_sum_requires_elementwise_check,
    )

    assert callable(tree_missing_values_estimator_name)
    assert callable(tree_missing_values_common_kwargs)
    assert callable(tree_missing_values_overall_sum_requires_elementwise_check)
    assert callable(tree_missing_values_overall_sum_has_missing)


def test_tree_missing_value_mask_prelude_matches_sklearn_logic() -> None:
    from sciona.atoms.ml.sklearn.tree.missing_value_mask_prelude import (
        tree_missing_values_common_kwargs,
        tree_missing_values_estimator_name,
        tree_missing_values_overall_sum_has_missing,
        tree_missing_values_overall_sum_requires_elementwise_check,
    )

    assert tree_missing_values_estimator_name("DecisionTreeClassifier") == "DecisionTreeClassifier"
    assert tree_missing_values_estimator_name("DecisionTreeClassifier", "CustomTree") == "CustomTree"
    assert tree_missing_values_common_kwargs("DecisionTreeClassifier") == {
        "estimator_name": "DecisionTreeClassifier",
        "input_name": "X",
    }

    assert tree_missing_values_overall_sum_requires_elementwise_check(1.0) is False
    assert tree_missing_values_overall_sum_requires_elementwise_check(math.inf) is True
    assert tree_missing_values_overall_sum_has_missing(1.0) is False
    assert tree_missing_values_overall_sum_has_missing(math.nan) is True


def test_tree_missing_value_mask_prelude_contracts() -> None:
    from sciona.atoms.ml.sklearn.tree.missing_value_mask_prelude import (
        tree_missing_values_estimator_name,
        tree_missing_values_overall_sum_has_missing,
    )

    with pytest.raises(ViolationError):
        tree_missing_values_estimator_name("", None)

    with pytest.raises(ViolationError):
        tree_missing_values_overall_sum_has_missing(True)

