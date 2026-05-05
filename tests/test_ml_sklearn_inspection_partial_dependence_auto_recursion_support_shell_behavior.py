from __future__ import annotations

import pytest
from icontract import ViolationError


def test_partial_dependence_auto_recursion_support_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.inspection.partial_dependence_auto_recursion_support_shell import (
        partial_dependence_gradient_boosting_recursion_supported,
        partial_dependence_recursion_supported_estimator,
        partial_dependence_tree_recursion_supported,
    )

    assert callable(partial_dependence_gradient_boosting_recursion_supported)
    assert callable(partial_dependence_tree_recursion_supported)
    assert callable(partial_dependence_recursion_supported_estimator)


def test_partial_dependence_auto_recursion_support_shell_matches_sklearn_logic() -> None:
    from sciona.atoms.ml.sklearn.inspection.partial_dependence_auto_recursion_support_shell import (
        partial_dependence_gradient_boosting_recursion_supported,
        partial_dependence_recursion_supported_estimator,
        partial_dependence_tree_recursion_supported,
    )

    assert partial_dependence_gradient_boosting_recursion_supported(
        is_base_gradient_boosting=True,
        init_is_none=True,
    ) is True
    assert partial_dependence_gradient_boosting_recursion_supported(
        is_base_gradient_boosting=True,
        init_is_none=False,
    ) is False

    assert partial_dependence_tree_recursion_supported(
        is_base_hist_gradient_boosting=False,
        is_decision_tree_regressor=False,
        is_random_forest_regressor=False,
    ) is False
    assert partial_dependence_tree_recursion_supported(
        is_base_hist_gradient_boosting=True,
        is_decision_tree_regressor=False,
        is_random_forest_regressor=False,
    ) is True

    assert partial_dependence_recursion_supported_estimator(
        gradient_boosting_supported=False,
        tree_supported=False,
    ) is False
    assert partial_dependence_recursion_supported_estimator(
        gradient_boosting_supported=True,
        tree_supported=False,
    ) is True
    assert partial_dependence_recursion_supported_estimator(
        gradient_boosting_supported=False,
        tree_supported=True,
    ) is True


def test_partial_dependence_auto_recursion_support_shell_contracts() -> None:
    from sciona.atoms.ml.sklearn.inspection.partial_dependence_auto_recursion_support_shell import (
        partial_dependence_gradient_boosting_recursion_supported,
    )

    with pytest.raises(ViolationError):
        partial_dependence_gradient_boosting_recursion_supported(
            is_base_gradient_boosting=True,
            init_is_none=1,
        )
