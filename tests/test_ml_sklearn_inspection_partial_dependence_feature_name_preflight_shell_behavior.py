from __future__ import annotations

import pytest
from icontract import ViolationError


def test_partial_dependence_feature_name_preflight_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.inspection.partial_dependence_feature_name_preflight_shell import (
        partial_dependence_feature_key_is_string,
        partial_dependence_feature_name_missing_guard_required,
        partial_dependence_feature_name_missing_message,
        partial_dependence_feature_names_required_guard_required,
        partial_dependence_feature_names_required_message,
    )

    assert callable(partial_dependence_feature_key_is_string)
    assert callable(partial_dependence_feature_names_required_guard_required)
    assert callable(partial_dependence_feature_names_required_message)
    assert callable(partial_dependence_feature_name_missing_guard_required)
    assert callable(partial_dependence_feature_name_missing_message)


def test_partial_dependence_feature_name_preflight_shell_matches_sklearn_logic() -> None:
    from sciona.atoms.ml.sklearn.inspection.partial_dependence_feature_name_preflight_shell import (
        partial_dependence_feature_key_is_string,
        partial_dependence_feature_name_missing_guard_required,
        partial_dependence_feature_name_missing_message,
        partial_dependence_feature_names_required_guard_required,
        partial_dependence_feature_names_required_message,
    )

    assert partial_dependence_feature_key_is_string("age") is True
    assert partial_dependence_feature_key_is_string(2) is False

    assert partial_dependence_feature_names_required_guard_required(True, False) is True
    assert partial_dependence_feature_names_required_guard_required(True, True) is False
    assert partial_dependence_feature_names_required_guard_required(False, False) is False
    assert partial_dependence_feature_names_required_message("age") == (
        "Cannot plot partial dependence for feature 'age' since "
        "the list of feature names was not provided, neither as "
        "column names of a pandas data-frame nor via the feature_names "
        "parameter."
    )

    feature_names = ("age", "zip_code", "income")
    assert partial_dependence_feature_name_missing_guard_required("state", feature_names) is True
    assert partial_dependence_feature_name_missing_guard_required("income", feature_names) is False
    assert partial_dependence_feature_name_missing_message("state") == "Feature 'state' not in feature_names"


def test_partial_dependence_feature_name_preflight_shell_contracts() -> None:
    from sciona.atoms.ml.sklearn.inspection.partial_dependence_feature_name_preflight_shell import (
        partial_dependence_feature_key_is_string,
        partial_dependence_feature_name_missing_guard_required,
        partial_dependence_feature_names_required_message,
    )

    with pytest.raises(ViolationError):
        partial_dependence_feature_key_is_string(True)

    with pytest.raises(ViolationError):
        partial_dependence_feature_names_required_message("")

    with pytest.raises(ViolationError):
        partial_dependence_feature_name_missing_guard_required("age", ())
