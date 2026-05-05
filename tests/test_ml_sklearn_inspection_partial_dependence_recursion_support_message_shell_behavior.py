from __future__ import annotations

import pytest
from icontract import ViolationError


def test_partial_dependence_recursion_support_message_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.inspection.partial_dependence_recursion_support_message_shell import (
        partial_dependence_recursion_support_guard_required,
        partial_dependence_supported_recursion_classes,
        partial_dependence_unsupported_recursion_message,
    )

    assert callable(partial_dependence_recursion_support_guard_required)
    assert callable(partial_dependence_supported_recursion_classes)
    assert callable(partial_dependence_unsupported_recursion_message)


def test_partial_dependence_recursion_support_message_shell_matches_sklearn_logic() -> None:
    from sciona.atoms.ml.sklearn.inspection.partial_dependence_recursion_support_message_shell import (
        partial_dependence_recursion_support_guard_required,
        partial_dependence_supported_recursion_classes,
        partial_dependence_unsupported_recursion_message,
    )

    assert partial_dependence_recursion_support_guard_required(
        "recursion",
        supports_recursion=False,
    ) is True
    assert partial_dependence_recursion_support_guard_required(
        "brute",
        supports_recursion=False,
    ) is False

    supported = partial_dependence_supported_recursion_classes("recursion")
    assert supported == (
        "GradientBoostingClassifier",
        "GradientBoostingRegressor",
        "HistGradientBoostingClassifier",
        "HistGradientBoostingRegressor",
        "HistGradientBoostingRegressor",
        "DecisionTreeRegressor",
        "RandomForestRegressor",
    )
    assert (
        partial_dependence_unsupported_recursion_message(supported)
        == "Only the following estimators support the 'recursion' method: "
        "GradientBoostingClassifier, GradientBoostingRegressor, "
        "HistGradientBoostingClassifier, HistGradientBoostingRegressor, "
        "HistGradientBoostingRegressor, DecisionTreeRegressor, "
        "RandomForestRegressor. Try using method='brute'."
    )


def test_partial_dependence_recursion_support_message_shell_contracts() -> None:
    from sciona.atoms.ml.sklearn.inspection.partial_dependence_recursion_support_message_shell import (
        partial_dependence_recursion_support_guard_required,
        partial_dependence_unsupported_recursion_message,
    )

    with pytest.raises(ViolationError):
        partial_dependence_recursion_support_guard_required("other", supports_recursion=False)

    with pytest.raises(ViolationError):
        partial_dependence_unsupported_recursion_message(tuple())
