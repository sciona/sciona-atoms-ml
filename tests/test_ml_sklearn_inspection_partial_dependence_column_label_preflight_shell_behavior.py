from __future__ import annotations

import pytest
from icontract import ViolationError


def test_partial_dependence_column_label_preflight_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.inspection.partial_dependence_column_label_preflight_shell import (
        partial_dependence_column_key_uses_label_branch,
        partial_dependence_dataframe_columns_required_guard_required,
        partial_dependence_dataframe_columns_required_message,
        partial_dependence_string_column_keys,
    )

    assert callable(partial_dependence_column_key_uses_label_branch)
    assert callable(partial_dependence_dataframe_columns_required_guard_required)
    assert callable(partial_dependence_dataframe_columns_required_message)
    assert callable(partial_dependence_string_column_keys)


def test_partial_dependence_column_label_preflight_shell_matches_sklearn_logic() -> None:
    from sciona.atoms.ml.sklearn.inspection.partial_dependence_column_label_preflight_shell import (
        partial_dependence_column_key_uses_label_branch,
        partial_dependence_dataframe_columns_required_guard_required,
        partial_dependence_dataframe_columns_required_message,
        partial_dependence_string_column_keys,
    )

    assert partial_dependence_column_key_uses_label_branch("str") is True
    assert partial_dependence_column_key_uses_label_branch("int") is False
    assert partial_dependence_column_key_uses_label_branch("bool") is False
    assert partial_dependence_column_key_uses_label_branch("none") is True

    assert partial_dependence_dataframe_columns_required_guard_required(True, False) is True
    assert partial_dependence_dataframe_columns_required_guard_required(True, True) is False
    assert partial_dependence_dataframe_columns_required_guard_required(False, False) is False
    assert partial_dependence_dataframe_columns_required_message(True) == (
        "Specifying the columns using strings is only supported for dataframes."
    )

    assert partial_dependence_string_column_keys("age") == ("age",)


def test_partial_dependence_column_label_preflight_shell_contracts() -> None:
    from sciona.atoms.ml.sklearn.inspection.partial_dependence_column_label_preflight_shell import (
        partial_dependence_column_key_uses_label_branch,
        partial_dependence_dataframe_columns_required_guard_required,
        partial_dependence_string_column_keys,
    )

    with pytest.raises(ViolationError):
        partial_dependence_column_key_uses_label_branch("object")

    with pytest.raises(ViolationError):
        partial_dependence_dataframe_columns_required_guard_required(True, 1)

    with pytest.raises(ViolationError):
        partial_dependence_string_column_keys("")
