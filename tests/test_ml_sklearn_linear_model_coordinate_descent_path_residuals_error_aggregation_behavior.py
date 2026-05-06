from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError


def test_coordinate_descent_path_residuals_error_aggregation_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_path_residuals_error_aggregation import (
        cd_path_residuals_intercepts,
        cd_path_residuals_mean_mse,
        cd_path_residuals_mse,
        cd_path_residuals_residues,
        cd_path_residuals_use_weighted_mse,
    )

    assert callable(cd_path_residuals_intercepts)
    assert callable(cd_path_residuals_residues)
    assert callable(cd_path_residuals_use_weighted_mse)
    assert callable(cd_path_residuals_mse)
    assert callable(cd_path_residuals_mean_mse)


def test_coordinate_descent_path_residuals_error_aggregation_matches_sklearn_shells() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_path_residuals_error_aggregation import (
        cd_path_residuals_intercepts,
        cd_path_residuals_mean_mse,
        cd_path_residuals_mse,
        cd_path_residuals_residues,
        cd_path_residuals_use_weighted_mse,
    )

    y_offset = np.array([1.5], dtype=np.float64)
    X_offset = np.array([0.5, -0.5], dtype=np.float64)
    coefs = np.array([[[2.0, 1.0], [1.0, 0.0]]], dtype=np.float64)
    intercepts = cd_path_residuals_intercepts(y_offset, X_offset, coefs)

    X_test_coefs = np.array(
        [
            [[3.0, 1.5]],
            [[4.0, 2.0]],
        ],
        dtype=np.float64,
    )
    y_test = np.array([[2.0], [3.0]], dtype=np.float64)
    residues = cd_path_residuals_residues(X_test_coefs, y_test, intercepts)

    assert cd_path_residuals_use_weighted_mse(None) is False
    mse = cd_path_residuals_mse(residues, None, False)
    mean_mse = cd_path_residuals_mean_mse(mse)

    assert intercepts.shape == (1, 2)
    assert residues.shape == (2, 1, 2)
    assert mse.shape == (1, 2)
    assert mean_mse.shape == (2,)

    sw_test = np.array([1.0, 3.0], dtype=np.float64)
    assert cd_path_residuals_use_weighted_mse(sw_test) is True
    weighted_mse = cd_path_residuals_mse(residues, sw_test, True)
    expected_weighted = np.average(residues**2, weights=sw_test, axis=0)
    assert np.allclose(weighted_mse, expected_weighted)


def test_coordinate_descent_path_residuals_error_aggregation_contracts() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_path_residuals_error_aggregation import (
        cd_path_residuals_intercepts,
        cd_path_residuals_mse,
    )

    with pytest.raises(ViolationError):
        cd_path_residuals_intercepts(
            np.array([1.0], dtype=np.float64),
            np.array([1.0], dtype=np.float64),
            np.array([[1.0, 2.0]], dtype=np.float64),
        )

    with pytest.raises(ViolationError):
        cd_path_residuals_mse(np.ones((2, 1, 2)), np.array([1.0]), True)
