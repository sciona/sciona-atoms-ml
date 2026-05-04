from __future__ import annotations

import pytest
from icontract import ViolationError


def test_partial_dependence_integer_warning_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.inspection.partial_dependence_integer_warning_shell import (
        partial_dependence_first_integer_warning_feature,
        partial_dependence_integer_warning_message,
        partial_dependence_integer_warning_required,
    )

    assert callable(partial_dependence_integer_warning_required)
    assert callable(partial_dependence_integer_warning_message)
    assert callable(partial_dependence_first_integer_warning_feature)


def test_partial_dependence_integer_warning_shell_matches_sklearn_logic() -> None:
    from sciona.atoms.ml.sklearn.inspection.partial_dependence_integer_warning_shell import (
        partial_dependence_first_integer_warning_feature,
        partial_dependence_integer_warning_message,
        partial_dependence_integer_warning_required,
    )

    assert partial_dependence_integer_warning_required(False, "i") is True
    assert partial_dependence_integer_warning_required(False, "u") is True
    assert partial_dependence_integer_warning_required(False, "f") is False
    assert partial_dependence_integer_warning_required(True, "i") is False

    features = ("age", "zip_code", "segment")
    is_categorical = (False, True, False)
    dtype_kinds = ("f", "i", "u")
    first_feature = partial_dependence_first_integer_warning_feature(features, is_categorical, dtype_kinds)
    assert first_feature == "segment"

    message = partial_dependence_integer_warning_message(first_feature)
    assert message == (
        "The column 'segment' contains integer data. Partial "
        "dependence plots are not supported for integer data: this "
        "can lead to implicit rounding with NumPy arrays or even errors "
        "with newer pandas versions. Please convert numerical features"
        "to floating point dtypes ahead of time to avoid problems. "
        "This will raise ValueError in scikit-learn 1.9."
    )

    assert partial_dependence_first_integer_warning_feature(("age",), (False,), ("f",)) is None


def test_partial_dependence_integer_warning_shell_contracts() -> None:
    from sciona.atoms.ml.sklearn.inspection.partial_dependence_integer_warning_shell import (
        partial_dependence_first_integer_warning_feature,
        partial_dependence_integer_warning_message,
        partial_dependence_integer_warning_required,
    )

    with pytest.raises(ViolationError):
        partial_dependence_integer_warning_required(0, "i")

    with pytest.raises(ViolationError):
        partial_dependence_integer_warning_message(["age"])

    with pytest.raises(ViolationError):
        partial_dependence_first_integer_warning_feature(("age",), (False, True), ("i",))
