from __future__ import annotations

import pytest
from icontract import ViolationError


def test_partial_dependence_feature_name_setup_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.inspection.partial_dependence_feature_name_setup_shell import (
        partial_dependence_default_feature_names,
        partial_dependence_duplicate_feature_names_guard_required,
        partial_dependence_duplicate_feature_names_message,
        partial_dependence_use_column_names_tolist,
        partial_dependence_use_feature_names_tolist,
    )

    assert callable(partial_dependence_use_column_names_tolist)
    assert callable(partial_dependence_default_feature_names)
    assert callable(partial_dependence_use_feature_names_tolist)
    assert callable(partial_dependence_duplicate_feature_names_guard_required)
    assert callable(partial_dependence_duplicate_feature_names_message)


def test_partial_dependence_feature_name_setup_shell_matches_sklearn_logic() -> None:
    from sciona.atoms.ml.sklearn.inspection.partial_dependence_feature_name_setup_shell import (
        partial_dependence_default_feature_names,
        partial_dependence_duplicate_feature_names_guard_required,
        partial_dependence_duplicate_feature_names_message,
        partial_dependence_use_column_names_tolist,
        partial_dependence_use_feature_names_tolist,
    )

    assert partial_dependence_use_column_names_tolist(True, True) is True
    assert partial_dependence_use_column_names_tolist(True, False) is False
    assert partial_dependence_use_column_names_tolist(False, True) is False

    assert partial_dependence_default_feature_names(0) == ()
    assert partial_dependence_default_feature_names(3) == ("x0", "x1", "x2")

    assert partial_dependence_use_feature_names_tolist(True, True) is True
    assert partial_dependence_use_feature_names_tolist(True, False) is False
    assert partial_dependence_use_feature_names_tolist(False, True) is False

    assert partial_dependence_duplicate_feature_names_guard_required(("age", "zip")) is False
    assert partial_dependence_duplicate_feature_names_guard_required(("age", "zip", "age")) is True
    assert partial_dependence_duplicate_feature_names_message(("age", "age")) == (
        "feature_names should not contain duplicates."
    )


def test_partial_dependence_feature_name_setup_shell_contracts() -> None:
    from sciona.atoms.ml.sklearn.inspection.partial_dependence_feature_name_setup_shell import (
        partial_dependence_default_feature_names,
        partial_dependence_duplicate_feature_names_guard_required,
        partial_dependence_use_column_names_tolist,
    )

    with pytest.raises(ViolationError):
        partial_dependence_use_column_names_tolist(True, 1)

    with pytest.raises(ViolationError):
        partial_dependence_default_feature_names(-1)

    with pytest.raises(ViolationError):
        partial_dependence_duplicate_feature_names_guard_required(("age", ""))
