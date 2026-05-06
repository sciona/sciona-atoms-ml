from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError


def test_coordinate_descent_cv_mse_selection_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_cv_mse_selection_shell import (
        cd_cv_alphas_from_auto_grid,
        cd_cv_alphas_from_user_grid,
        cd_cv_best_alpha_index,
        cd_cv_best_alpha_value,
        cd_cv_best_l1_ratio_value,
        cd_cv_best_mse_value,
        cd_cv_mean_mse,
        cd_cv_mse_path_public,
        cd_cv_mse_paths_reshaped,
    )

    assert callable(cd_cv_mse_paths_reshaped)
    assert callable(cd_cv_mean_mse)
    assert callable(cd_cv_mse_path_public)
    assert callable(cd_cv_best_alpha_index)
    assert callable(cd_cv_best_mse_value)
    assert callable(cd_cv_best_alpha_value)
    assert callable(cd_cv_best_l1_ratio_value)
    assert callable(cd_cv_alphas_from_auto_grid)
    assert callable(cd_cv_alphas_from_user_grid)


def test_coordinate_descent_cv_mse_selection_shell_matches_sklearn_shells() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_cv_mse_selection_shell import (
        cd_cv_alphas_from_auto_grid,
        cd_cv_alphas_from_user_grid,
        cd_cv_best_alpha_index,
        cd_cv_best_alpha_value,
        cd_cv_best_l1_ratio_value,
        cd_cv_best_mse_value,
        cd_cv_mean_mse,
        cd_cv_mse_path_public,
        cd_cv_mse_paths_reshaped,
    )

    flat = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    reshaped = cd_cv_mse_paths_reshaped(flat, 1, 2)
    assert reshaped.shape == (1, 2, 3)
    mean_mse = cd_cv_mean_mse(reshaped)
    assert np.array_equal(mean_mse, np.array([[2.5, 3.5, 4.5]]))
    public = cd_cv_mse_path_public(reshaped)
    assert np.array_equal(public, np.array([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]))
    mse_alphas = np.array([0.3, 0.1, 0.2])
    idx = cd_cv_best_alpha_index(mse_alphas)
    assert idx == 1
    assert cd_cv_best_mse_value(mse_alphas, idx) == pytest.approx(0.1)
    l1_alphas = np.array([10.0, 5.0, 1.0])
    assert cd_cv_best_alpha_value(l1_alphas, idx) == pytest.approx(5.0)
    token = object()
    assert cd_cv_best_l1_ratio_value(token) is token
    assert np.array_equal(cd_cv_alphas_from_auto_grid([[3.0, 2.0, 1.0]], 1), np.array([3.0, 2.0, 1.0]))
    assert np.array_equal(cd_cv_alphas_from_user_grid([[3.0, 2.0, 1.0]]), np.array([3.0, 2.0, 1.0]))


def test_coordinate_descent_cv_mse_selection_shell_contracts() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_cv_mse_selection_shell import (
        cd_cv_best_alpha_index,
        cd_cv_mse_paths_reshaped,
    )

    with pytest.raises(ViolationError):
        cd_cv_best_alpha_index(np.array([]))

    with pytest.raises(ViolationError):
        cd_cv_mse_paths_reshaped(np.array([1.0]), 0, 1)
