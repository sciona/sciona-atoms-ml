from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError


def test_coordinate_descent_path_residuals_callback_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_path_residuals_callback_shell import (
        cd_path_residuals_check_array_accept_sparse,
        cd_path_residuals_check_array_dtype,
        cd_path_residuals_check_array_order,
        cd_path_residuals_path_result_alphas,
        cd_path_residuals_path_result_coefs,
    )

    assert callable(cd_path_residuals_check_array_accept_sparse)
    assert callable(cd_path_residuals_check_array_dtype)
    assert callable(cd_path_residuals_check_array_order)
    assert callable(cd_path_residuals_path_result_alphas)
    assert callable(cd_path_residuals_path_result_coefs)


def test_coordinate_descent_path_residuals_callback_shell_matches_sklearn_shells() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_path_residuals_callback_shell import (
        cd_path_residuals_check_array_accept_sparse,
        cd_path_residuals_check_array_dtype,
        cd_path_residuals_check_array_order,
        cd_path_residuals_path_result_alphas,
        cd_path_residuals_path_result_coefs,
    )

    dtype = np.float64
    order = "F"
    path_result = (np.array([1.0, 0.5]), np.array([[3.0, 2.0], [1.0, 0.0]]), {"n_iter": [2, 3]})

    assert cd_path_residuals_check_array_accept_sparse(dtype) == "csc"
    assert cd_path_residuals_check_array_dtype(dtype) is dtype
    assert cd_path_residuals_check_array_order(order) is order
    assert cd_path_residuals_path_result_alphas(path_result) is path_result[0]
    assert cd_path_residuals_path_result_coefs(path_result) is path_result[1]


def test_coordinate_descent_path_residuals_callback_shell_contracts() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_path_residuals_callback_shell import (
        cd_path_residuals_path_result_alphas,
        cd_path_residuals_path_result_coefs,
    )

    with pytest.raises(ViolationError):
        cd_path_residuals_path_result_alphas((np.array([1.0]), np.array([[2.0]])))

    with pytest.raises(ViolationError):
        cd_path_residuals_path_result_coefs("abc")
