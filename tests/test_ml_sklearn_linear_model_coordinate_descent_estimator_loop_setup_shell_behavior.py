from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError


def test_coordinate_descent_estimator_loop_setup_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_estimator_loop_setup_shell import (
        cd_estimator_dual_gaps_zeros,
        cd_estimator_initial_coef_required,
        cd_estimator_initial_coef_zeros,
        cd_estimator_loop_this_xy,
        cd_estimator_n_iter_list_initial,
        cd_estimator_path_args,
        cd_estimator_path_kwargs,
        cd_estimator_single_alpha_grid,
        cd_estimator_warm_start_coef_matrix,
    )

    assert callable(cd_estimator_initial_coef_required)
    assert callable(cd_estimator_initial_coef_zeros)
    assert callable(cd_estimator_warm_start_coef_matrix)
    assert callable(cd_estimator_dual_gaps_zeros)
    assert callable(cd_estimator_n_iter_list_initial)
    assert callable(cd_estimator_loop_this_xy)
    assert callable(cd_estimator_single_alpha_grid)
    assert callable(cd_estimator_path_args)
    assert callable(cd_estimator_path_kwargs)


def test_coordinate_descent_estimator_loop_setup_shell_matches_sklearn_setup() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_estimator_loop_setup_shell import (
        cd_estimator_dual_gaps_zeros,
        cd_estimator_initial_coef_required,
        cd_estimator_initial_coef_zeros,
        cd_estimator_loop_this_xy,
        cd_estimator_n_iter_list_initial,
        cd_estimator_path_args,
        cd_estimator_path_kwargs,
        cd_estimator_single_alpha_grid,
        cd_estimator_warm_start_coef_matrix,
    )

    assert cd_estimator_initial_coef_required(False, True) is True
    assert cd_estimator_initial_coef_required(True, False) is True
    assert cd_estimator_initial_coef_required(True, True) is False

    coef = cd_estimator_initial_coef_zeros(2, 3, np.float64)
    assert coef.shape == (2, 3)
    assert coef.flags["F_CONTIGUOUS"]
    assert np.count_nonzero(coef) == 0

    vector_coef = np.array([1.0, 2.0, 3.0])
    assert np.array_equal(cd_estimator_warm_start_coef_matrix(vector_coef), vector_coef[np.newaxis, :])
    matrix_coef = np.array([[1.0, 2.0]])
    assert cd_estimator_warm_start_coef_matrix(matrix_coef) is matrix_coef

    dual_gaps = cd_estimator_dual_gaps_zeros(2, np.float32)
    assert dual_gaps.shape == (2,)
    assert dual_gaps.dtype == np.float32
    assert cd_estimator_n_iter_list_initial(2) == []

    Xy = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert np.array_equal(cd_estimator_loop_this_xy(Xy, 1), Xy[:, 1])
    assert cd_estimator_loop_this_xy(None, 0) is None

    alpha = object()
    alphas = cd_estimator_single_alpha_grid(alpha)
    assert alphas == [alpha]
    assert alphas[0] is alpha

    X = object()
    y = np.array([[1.0, 2.0], [3.0, 4.0]])
    path_args = cd_estimator_path_args(X, y, 0)
    assert path_args[0] is X
    assert np.array_equal(path_args[1], y[:, 0])

    kwargs = cd_estimator_path_kwargs(
        0.5,
        1.0,
        "auto",
        None,
        coef[0],
        False,
        1e-4,
        np.array([0.0, 0.0, 0.0]),
        np.ones(3),
        1000,
        123,
        "cyclic",
        None,
    )
    assert kwargs["eps"] is None
    assert kwargs["n_alphas"] is None
    assert kwargs["alphas"] == [1.0]
    assert kwargs["copy_X"] is True
    assert kwargs["verbose"] is False
    assert kwargs["return_n_iter"] is True
    assert kwargs["check_input"] is False
    assert kwargs["selection"] == "cyclic"


def test_coordinate_descent_estimator_loop_setup_shell_contracts() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_estimator_loop_setup_shell import (
        cd_estimator_dual_gaps_zeros,
        cd_estimator_initial_coef_required,
        cd_estimator_initial_coef_zeros,
        cd_estimator_loop_this_xy,
        cd_estimator_path_args,
        cd_estimator_path_kwargs,
        cd_estimator_warm_start_coef_matrix,
    )

    with pytest.raises(ViolationError):
        cd_estimator_initial_coef_required(True, "yes")  # type: ignore[arg-type]

    with pytest.raises(ViolationError):
        cd_estimator_initial_coef_zeros(0, 2, np.float64)

    with pytest.raises(ViolationError):
        cd_estimator_warm_start_coef_matrix(np.array([[[1.0]]]))

    with pytest.raises(ViolationError):
        cd_estimator_dual_gaps_zeros(1, object())

    with pytest.raises(ViolationError):
        cd_estimator_loop_this_xy(np.ones((2, 1)), 2)

    with pytest.raises(ViolationError):
        cd_estimator_path_args(object(), np.ones((2, 1)), 1)

    with pytest.raises(ViolationError):
        cd_estimator_path_kwargs(0.5, 1.0, False, None, np.zeros(2), False, 1e-4, None, None, 0, None, "cyclic", None)
