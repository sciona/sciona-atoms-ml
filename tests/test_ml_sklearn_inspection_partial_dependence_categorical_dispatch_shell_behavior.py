from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError


def test_partial_dependence_categorical_dispatch_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.inspection.partial_dependence_categorical_dispatch_shell import (
        partial_dependence_categorical_array,
        partial_dependence_categorical_bool_branch,
        partial_dependence_categorical_index_or_name_branch,
    )

    assert callable(partial_dependence_categorical_array)
    assert callable(partial_dependence_categorical_bool_branch)
    assert callable(partial_dependence_categorical_index_or_name_branch)


def test_partial_dependence_categorical_dispatch_shell_matches_sklearn_logic() -> None:
    from sciona.atoms.ml.sklearn.inspection.partial_dependence_categorical_dispatch_shell import (
        partial_dependence_categorical_array,
        partial_dependence_categorical_bool_branch,
        partial_dependence_categorical_index_or_name_branch,
    )

    bool_array = partial_dependence_categorical_array([True, False, True])
    assert isinstance(bool_array, np.ndarray)
    assert bool_array.dtype.kind == "b"

    str_array = partial_dependence_categorical_array(["city", "zip"])
    assert isinstance(str_array, np.ndarray)
    assert str_array.dtype.kind in {"U", "O"}

    assert partial_dependence_categorical_bool_branch("b") is True
    assert partial_dependence_categorical_bool_branch("i") is False

    assert partial_dependence_categorical_index_or_name_branch("i") is True
    assert partial_dependence_categorical_index_or_name_branch("O") is True
    assert partial_dependence_categorical_index_or_name_branch("U") is True
    assert partial_dependence_categorical_index_or_name_branch("b") is False


def test_partial_dependence_categorical_dispatch_shell_contracts() -> None:
    from sciona.atoms.ml.sklearn.inspection.partial_dependence_categorical_dispatch_shell import (
        partial_dependence_categorical_array,
        partial_dependence_categorical_bool_branch,
    )

    with pytest.raises(ViolationError):
        partial_dependence_categorical_array([])

    with pytest.raises(ViolationError):
        partial_dependence_categorical_bool_branch("")
