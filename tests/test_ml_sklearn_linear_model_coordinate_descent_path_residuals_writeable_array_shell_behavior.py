from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError


def test_coordinate_descent_path_residuals_writeable_array_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_path_residuals_writeable_array_shell import (
        cd_path_residuals_array_needs_writeable_fix,
        cd_path_residuals_dense_writeable_guard,
        cd_path_residuals_writable_array,
    )

    assert callable(cd_path_residuals_dense_writeable_guard)
    assert callable(cd_path_residuals_array_needs_writeable_fix)
    assert callable(cd_path_residuals_writable_array)


def test_coordinate_descent_path_residuals_writeable_array_shell_matches_sklearn_shells() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_path_residuals_writeable_array_shell import (
        cd_path_residuals_array_needs_writeable_fix,
        cd_path_residuals_dense_writeable_guard,
        cd_path_residuals_writable_array,
    )

    assert cd_path_residuals_dense_writeable_guard(False) is True
    assert cd_path_residuals_dense_writeable_guard(True) is False
    assert cd_path_residuals_array_needs_writeable_fix(False, False) is True
    assert cd_path_residuals_array_needs_writeable_fix(True, False) is False
    assert cd_path_residuals_array_needs_writeable_fix(False, True) is False

    array_input = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float64)
    writeable_view = array_input[1:3]
    same = cd_path_residuals_writable_array(writeable_view, array_input)
    assert same is writeable_view
    assert same.flags["WRITEABLE"]

    readonly_base = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    readonly_view = readonly_base[1:3]
    readonly_view.setflags(write=False)
    fixed = cd_path_residuals_writable_array(readonly_view, object())
    assert fixed is readonly_view
    assert fixed.flags["WRITEABLE"]
    assert np.array_equal(fixed, np.array([2.0, 3.0], dtype=np.float64))


def test_coordinate_descent_path_residuals_writeable_array_shell_contracts() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_path_residuals_writeable_array_shell import (
        cd_path_residuals_array_needs_writeable_fix,
        cd_path_residuals_writable_array,
    )

    with pytest.raises(ViolationError):
        cd_path_residuals_array_needs_writeable_fix("no", False)

    with pytest.raises(ViolationError):
        cd_path_residuals_writable_array(np.array([1.0, np.nan], dtype=np.float64), object())
