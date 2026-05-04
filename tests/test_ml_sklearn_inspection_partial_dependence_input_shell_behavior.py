from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError
from scipy import sparse
from sklearn.utils import check_array


def test_partial_dependence_input_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.inspection.partial_dependence_input_shell import (
        partial_dependence_checked_object_array,
        partial_dependence_use_object_check_array,
    )

    assert callable(partial_dependence_use_object_check_array)
    assert callable(partial_dependence_checked_object_array)


def test_partial_dependence_input_shell_matches_sklearn_branch_logic() -> None:
    from sciona.atoms.ml.sklearn.inspection.partial_dependence_input_shell import (
        partial_dependence_checked_object_array,
        partial_dependence_use_object_check_array,
    )

    rows = [[1.0, "a"], [np.nan, "b"], [3.0, "c"]]
    expected = np.asarray(check_array(rows, ensure_all_finite="allow-nan", dtype=object), dtype=object)

    actual = partial_dependence_checked_object_array(rows)

    assert partial_dependence_use_object_check_array(has_array=False, is_sparse=False) is True
    assert partial_dependence_use_object_check_array(has_array=True, is_sparse=False) is False
    assert partial_dependence_use_object_check_array(has_array=False, is_sparse=True) is False
    assert actual.shape == expected.shape
    for lhs, rhs in zip(actual.ravel(), expected.ravel(), strict=True):
        if isinstance(lhs, float) and isinstance(rhs, float) and np.isnan(lhs) and np.isnan(rhs):
            continue
        assert lhs == rhs

    assert hasattr(np.asarray(rows), "__array__")
    assert sparse.issparse(sparse.csr_matrix(np.eye(2)))


def test_partial_dependence_input_shell_contracts() -> None:
    from sciona.atoms.ml.sklearn.inspection.partial_dependence_input_shell import (
        partial_dependence_checked_object_array,
        partial_dependence_use_object_check_array,
    )

    with pytest.raises(ViolationError):
        partial_dependence_use_object_check_array(1, False)

    with pytest.raises(ViolationError):
        partial_dependence_checked_object_array(np.array([1.0, 2.0], dtype=object))
