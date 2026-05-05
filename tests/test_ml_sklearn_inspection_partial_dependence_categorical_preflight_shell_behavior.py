from __future__ import annotations

import pytest
from icontract import ViolationError


def test_partial_dependence_categorical_preflight_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.inspection.partial_dependence_categorical_preflight_shell import (
        partial_dependence_categorical_bool_size_guard_required,
        partial_dependence_categorical_bool_size_message,
        partial_dependence_categorical_dtype_message,
        partial_dependence_categorical_dtype_supported,
        partial_dependence_categorical_empty_guard_required,
        partial_dependence_categorical_empty_message,
    )

    assert callable(partial_dependence_categorical_empty_guard_required)
    assert callable(partial_dependence_categorical_empty_message)
    assert callable(partial_dependence_categorical_bool_size_guard_required)
    assert callable(partial_dependence_categorical_bool_size_message)
    assert callable(partial_dependence_categorical_dtype_supported)
    assert callable(partial_dependence_categorical_dtype_message)


def test_partial_dependence_categorical_preflight_shell_matches_sklearn_logic() -> None:
    from sciona.atoms.ml.sklearn.inspection.partial_dependence_categorical_preflight_shell import (
        partial_dependence_categorical_bool_size_guard_required,
        partial_dependence_categorical_bool_size_message,
        partial_dependence_categorical_dtype_message,
        partial_dependence_categorical_dtype_supported,
        partial_dependence_categorical_empty_guard_required,
        partial_dependence_categorical_empty_message,
    )

    assert partial_dependence_categorical_empty_guard_required(0) is True
    assert partial_dependence_categorical_empty_guard_required(2) is False
    assert partial_dependence_categorical_empty_message(0) == (
        "Passing an empty list (`[]`) to `categorical_features` is not "
        "supported. Use `None` instead to indicate that there are no "
        "categorical features."
    )

    assert partial_dependence_categorical_bool_size_guard_required(3, 5) is True
    assert partial_dependence_categorical_bool_size_guard_required(5, 5) is False
    assert partial_dependence_categorical_bool_size_message(3, 5) == (
        "When `categorical_features` is a boolean array-like, "
        "the array should be of shape (n_features,). Got "
        "3 elements while `X` contains "
        "5 features."
    )

    assert partial_dependence_categorical_dtype_supported("b") is True
    assert partial_dependence_categorical_dtype_supported("i") is True
    assert partial_dependence_categorical_dtype_supported("O") is True
    assert partial_dependence_categorical_dtype_supported("U") is True
    assert partial_dependence_categorical_dtype_supported("f") is False
    assert partial_dependence_categorical_dtype_message("float64") == (
        "Expected `categorical_features` to be an array-like of boolean,"
        " integer, or string. Got float64 instead."
    )


def test_partial_dependence_categorical_preflight_shell_contracts() -> None:
    from sciona.atoms.ml.sklearn.inspection.partial_dependence_categorical_preflight_shell import (
        partial_dependence_categorical_bool_size_guard_required,
        partial_dependence_categorical_dtype_message,
        partial_dependence_categorical_empty_guard_required,
    )

    with pytest.raises(ViolationError):
        partial_dependence_categorical_empty_guard_required(-1)

    with pytest.raises(ViolationError):
        partial_dependence_categorical_bool_size_guard_required(1, 0)

    with pytest.raises(ViolationError):
        partial_dependence_categorical_dtype_message("")
