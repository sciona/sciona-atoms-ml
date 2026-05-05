from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError


def test_coordinate_descent_enet_path_input_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_enet_path_input_shell import (
        cd_enet_path_alpha_grid_required,
        cd_enet_path_check_input_branch,
        cd_enet_path_prefit_kwargs,
        cd_enet_path_sparse_scaling,
        cd_enet_path_sparse_scaling_required,
        cd_enet_path_unexpected_params_guard_required,
        cd_enet_path_Xy_validation_required,
    )

    assert callable(cd_enet_path_unexpected_params_guard_required)
    assert callable(cd_enet_path_check_input_branch)
    assert callable(cd_enet_path_Xy_validation_required)
    assert callable(cd_enet_path_sparse_scaling_required)
    assert callable(cd_enet_path_sparse_scaling)
    assert callable(cd_enet_path_prefit_kwargs)
    assert callable(cd_enet_path_alpha_grid_required)


def test_coordinate_descent_enet_path_input_shell_matches_sklearn_shells() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_enet_path_input_shell import (
        cd_enet_path_alpha_grid_required,
        cd_enet_path_check_input_branch,
        cd_enet_path_prefit_kwargs,
        cd_enet_path_sparse_scaling,
        cd_enet_path_sparse_scaling_required,
        cd_enet_path_unexpected_params_guard_required,
        cd_enet_path_Xy_validation_required,
    )

    assert cd_enet_path_unexpected_params_guard_required({"bad": 1}) is True
    assert cd_enet_path_check_input_branch(True) is True
    assert cd_enet_path_Xy_validation_required(np.array([1.0, 2.0], dtype=np.float64)) is True
    assert cd_enet_path_sparse_scaling_required(multi_output=False, x_is_sparse=True) is True
    assert np.allclose(
        cd_enet_path_sparse_scaling(
            X_offset_param=np.array([2.0, 6.0], dtype=np.float64),
            X_scale_param=np.array([2.0, 3.0], dtype=np.float64),
            n_features=2,
            dtype_name="float64",
        ),
        np.array([1.0, 2.0], dtype=np.float64),
    )
    assert np.array_equal(
        cd_enet_path_sparse_scaling(
            X_offset_param=None,
            X_scale_param=None,
            n_features=3,
            dtype_name="float32",
        ),
        np.zeros(3, dtype=np.float32),
    )
    assert cd_enet_path_prefit_kwargs(False) == {
        "fit_intercept": False,
        "copy": False,
        "check_input": False,
    }
    assert cd_enet_path_alpha_grid_required(None) is True


def test_coordinate_descent_enet_path_input_shell_contracts() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_enet_path_input_shell import (
        cd_enet_path_check_input_branch,
        cd_enet_path_sparse_scaling,
        cd_enet_path_unexpected_params_guard_required,
    )

    with pytest.raises(ViolationError):
        cd_enet_path_unexpected_params_guard_required([("bad", 1)])

    with pytest.raises(ViolationError):
        cd_enet_path_check_input_branch(1)

    with pytest.raises(ViolationError):
        cd_enet_path_sparse_scaling(
            X_offset_param=np.array([1.0], dtype=np.float64),
            X_scale_param=np.array([0.0], dtype=np.float64),
            n_features=1,
            dtype_name="float64",
        )
