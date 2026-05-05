from __future__ import annotations

import pytest
from icontract import ViolationError


def test_partial_dependence_brute_response_method_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.inspection.partial_dependence_brute_response_method_shell import (
        partial_dependence_brute_auto_response_method,
        partial_dependence_brute_auto_target_method,
        partial_dependence_brute_resolved_response_method,
    )

    assert callable(partial_dependence_brute_auto_response_method)
    assert callable(partial_dependence_brute_auto_target_method)
    assert callable(partial_dependence_brute_resolved_response_method)


def test_partial_dependence_brute_response_method_shell_matches_sklearn_logic() -> None:
    from sciona.atoms.ml.sklearn.inspection.partial_dependence_brute_response_method_shell import (
        partial_dependence_brute_auto_response_method,
        partial_dependence_brute_auto_target_method,
        partial_dependence_brute_resolved_response_method,
    )

    assert partial_dependence_brute_auto_response_method("auto") is True
    assert partial_dependence_brute_auto_response_method("predict_proba") is False

    assert partial_dependence_brute_auto_target_method(
        is_regressor_task=True
    ) == "predict"
    assert partial_dependence_brute_auto_target_method(
        is_regressor_task=False
    ) == ("predict_proba", "decision_function")

    assert partial_dependence_brute_resolved_response_method(
        "auto",
        is_regressor_task=True,
    ) == "predict"
    assert partial_dependence_brute_resolved_response_method(
        "auto",
        is_regressor_task=False,
    ) == ("predict_proba", "decision_function")
    assert partial_dependence_brute_resolved_response_method(
        "decision_function",
        is_regressor_task=False,
    ) == "decision_function"


def test_partial_dependence_brute_response_method_shell_contracts() -> None:
    from sciona.atoms.ml.sklearn.inspection.partial_dependence_brute_response_method_shell import (
        partial_dependence_brute_auto_response_method,
        partial_dependence_brute_auto_target_method,
    )

    with pytest.raises(ViolationError):
        partial_dependence_brute_auto_response_method("predict")

    with pytest.raises(ViolationError):
        partial_dependence_brute_auto_target_method(is_regressor_task=1)
