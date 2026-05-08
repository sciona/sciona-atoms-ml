from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError
from scipy import sparse
from sklearn.utils.extmath import safe_sparse_dot


def test_coordinate_descent_path_residuals_projection_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_path_residuals_projection_shell import (
        cd_path_residuals_project_test_coefs,
    )

    assert callable(cd_path_residuals_project_test_coefs)


def test_coordinate_descent_path_residuals_projection_shell_matches_dense_safe_sparse_dot() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_path_residuals_projection_shell import (
        cd_path_residuals_project_test_coefs,
    )

    X_test = np.array([[1.0, 2.0], [-1.0, 0.5], [0.25, -0.75]], dtype=np.float64)
    coefs = np.array(
        [
            [[1.0, 0.5, -1.0], [2.0, -0.25, 0.0]],
            [[-0.5, 1.5, 2.0], [0.25, -1.0, 0.5]],
        ],
        dtype=np.float64,
    )

    projected = cd_path_residuals_project_test_coefs(X_test, coefs)
    expected = safe_sparse_dot(X_test, coefs)

    assert projected.shape == (3, 2, 3)
    assert np.allclose(projected, expected)


def test_coordinate_descent_path_residuals_projection_shell_matches_sparse_safe_sparse_dot() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_path_residuals_projection_shell import (
        cd_path_residuals_project_test_coefs,
    )

    X_test = sparse.csr_matrix(np.array([[0.0, 2.0, 0.0], [1.0, 0.0, -3.0]], dtype=np.float64))
    coefs = np.array(
        [
            [[1.0, 0.5], [2.0, -0.25], [0.0, 1.0]],
            [[-0.5, 1.5], [0.25, -1.0], [2.0, 0.5]],
        ],
        dtype=np.float64,
    )

    projected = cd_path_residuals_project_test_coefs(X_test, coefs)
    expected = safe_sparse_dot(X_test, coefs)

    assert projected.shape == (2, 2, 2)
    assert np.allclose(projected, expected)


def test_coordinate_descent_path_residuals_projection_shell_contracts() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_path_residuals_projection_shell import (
        cd_path_residuals_project_test_coefs,
    )

    with pytest.raises(ViolationError):
        cd_path_residuals_project_test_coefs(
            np.ones((2, 3), dtype=np.float64),
            np.ones((1, 2, 4), dtype=np.float64),
        )

    with pytest.raises(ViolationError):
        cd_path_residuals_project_test_coefs(
            np.ones((2, 3), dtype=np.float64),
            np.ones((3, 4), dtype=np.float64),
        )
