from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError


def test_coordinate_descent_path_residuals_path_params_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_path_residuals_path_params_shell import (
        cd_path_residuals_l1_ratio_update_required,
        cd_path_residuals_path_params_alphas,
        cd_path_residuals_path_params_copy_x,
        cd_path_residuals_path_params_l1_ratio,
        cd_path_residuals_path_params_precompute,
        cd_path_residuals_path_params_sample_weight,
        cd_path_residuals_path_params_X_offset,
        cd_path_residuals_path_params_X_scale,
        cd_path_residuals_path_params_Xy,
        cd_path_residuals_prefit_copy_flag,
    )

    assert callable(cd_path_residuals_prefit_copy_flag)
    assert callable(cd_path_residuals_path_params_Xy)
    assert callable(cd_path_residuals_path_params_X_offset)
    assert callable(cd_path_residuals_path_params_X_scale)
    assert callable(cd_path_residuals_path_params_precompute)
    assert callable(cd_path_residuals_path_params_copy_x)
    assert callable(cd_path_residuals_path_params_alphas)
    assert callable(cd_path_residuals_path_params_sample_weight)
    assert callable(cd_path_residuals_l1_ratio_update_required)
    assert callable(cd_path_residuals_path_params_l1_ratio)


def test_coordinate_descent_path_residuals_path_params_shell_matches_sklearn_shells() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_path_residuals_path_params_shell import (
        cd_path_residuals_l1_ratio_update_required,
        cd_path_residuals_path_params_alphas,
        cd_path_residuals_path_params_copy_x,
        cd_path_residuals_path_params_l1_ratio,
        cd_path_residuals_path_params_precompute,
        cd_path_residuals_path_params_sample_weight,
        cd_path_residuals_path_params_X_offset,
        cd_path_residuals_path_params_X_scale,
        cd_path_residuals_path_params_Xy,
        cd_path_residuals_prefit_copy_flag,
    )

    Xy = np.array([1.0, 2.0, 3.0])
    X_offset = np.array([0.25, -0.5])
    X_scale = np.array([1.0, 2.0])
    alphas = np.array([0.5, 0.1])
    sample_weight = np.array([0.7, 1.3])
    l1_ratio = 0.25

    path_params = {"precompute": "auto", "l1_ratio": 0.5}

    assert cd_path_residuals_prefit_copy_flag(True) is False
    assert cd_path_residuals_path_params_Xy(Xy) is Xy
    assert cd_path_residuals_path_params_X_offset(X_offset) is X_offset
    assert cd_path_residuals_path_params_X_scale(X_scale) is X_scale
    assert cd_path_residuals_path_params_precompute("auto") == "auto"
    assert cd_path_residuals_path_params_copy_x(path_params) is False
    assert cd_path_residuals_path_params_alphas(alphas) is alphas
    assert cd_path_residuals_path_params_sample_weight(sample_weight) is sample_weight
    assert cd_path_residuals_l1_ratio_update_required(path_params) is True
    assert cd_path_residuals_l1_ratio_update_required({"precompute": "auto"}) is False
    assert cd_path_residuals_path_params_l1_ratio(l1_ratio) is l1_ratio


def test_coordinate_descent_path_residuals_path_params_shell_contracts() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_path_residuals_path_params_shell import (
        cd_path_residuals_l1_ratio_update_required,
        cd_path_residuals_prefit_copy_flag,
    )

    with pytest.raises(ViolationError):
        cd_path_residuals_l1_ratio_update_required(["l1_ratio"])

    with pytest.raises(ViolationError):
        cd_path_residuals_prefit_copy_flag("yes")
