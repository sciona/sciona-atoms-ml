from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError


def test_coordinate_descent_multitask_solver_setup_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_multitask_solver_setup_shell import (
        cd_multitask_coef_fortran_array,
        cd_multitask_fresh_coef_required,
        cd_multitask_initial_coef_zeros,
        cd_multitask_preprocess_data_args,
        cd_multitask_preprocess_data_kwargs,
        cd_multitask_random_state_args,
        cd_multitask_regularization,
        cd_multitask_solver_args,
    )

    assert callable(cd_multitask_preprocess_data_args)
    assert callable(cd_multitask_preprocess_data_kwargs)
    assert callable(cd_multitask_fresh_coef_required)
    assert callable(cd_multitask_initial_coef_zeros)
    assert callable(cd_multitask_regularization)
    assert callable(cd_multitask_coef_fortran_array)
    assert callable(cd_multitask_random_state_args)
    assert callable(cd_multitask_solver_args)


def test_coordinate_descent_multitask_solver_setup_shell_matches_sklearn_setup() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_multitask_solver_setup_shell import (
        cd_multitask_coef_fortran_array,
        cd_multitask_fresh_coef_required,
        cd_multitask_initial_coef_zeros,
        cd_multitask_preprocess_data_args,
        cd_multitask_preprocess_data_kwargs,
        cd_multitask_random_state_args,
        cd_multitask_regularization,
        cd_multitask_solver_args,
    )

    X = np.asfortranarray(np.arange(6.0, dtype=np.float32).reshape(3, 2))
    y = np.asfortranarray(np.arange(9.0, dtype=np.float32).reshape(3, 3))

    preprocess_args = cd_multitask_preprocess_data_args(X, y)
    assert preprocess_args == (X, y)
    assert preprocess_args[0] is X
    assert preprocess_args[1] is y
    assert cd_multitask_preprocess_data_kwargs(True) == {
        "fit_intercept": True,
        "copy": False,
    }
    assert cd_multitask_preprocess_data_kwargs(False) == {
        "fit_intercept": False,
        "copy": False,
    }

    assert cd_multitask_fresh_coef_required(False, True) is True
    assert cd_multitask_fresh_coef_required(True, False) is True
    assert cd_multitask_fresh_coef_required(True, True) is False

    coef = cd_multitask_initial_coef_zeros(3, 2, np.float32)
    assert coef.shape == (3, 2)
    assert coef.dtype == np.float32
    assert coef.flags["F_CONTIGUOUS"]
    assert np.count_nonzero(coef) == 0

    assert cd_multitask_regularization(0.2, 0.75, 10) == pytest.approx((1.5, 0.5))

    c_order_coef = np.array([[1.0, 2.0], [3.0, 4.0]], order="C")
    fortran_coef = cd_multitask_coef_fortran_array(c_order_coef)
    assert fortran_coef.flags["F_CONTIGUOUS"]
    assert np.array_equal(fortran_coef, c_order_coef)

    random_state = object()
    random_state_args = cd_multitask_random_state_args(random_state)
    assert random_state_args == (random_state,)
    assert random_state_args[0] is random_state

    l1_reg, l2_reg = cd_multitask_regularization(0.2, 0.75, X.shape[0])
    checked_random_state = object()
    solver_args = cd_multitask_solver_args(
        fortran_coef,
        l1_reg,
        l2_reg,
        X,
        y,
        1000,
        1e-4,
        checked_random_state,
        True,
    )
    assert solver_args == (
        fortran_coef,
        l1_reg,
        l2_reg,
        X,
        y,
        1000,
        1e-4,
        checked_random_state,
        True,
    )
    assert solver_args[0] is fortran_coef
    assert solver_args[3] is X
    assert solver_args[4] is y
    assert solver_args[7] is checked_random_state


def test_coordinate_descent_multitask_solver_setup_shell_contracts() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_multitask_solver_setup_shell import (
        cd_multitask_coef_fortran_array,
        cd_multitask_fresh_coef_required,
        cd_multitask_initial_coef_zeros,
        cd_multitask_preprocess_data_kwargs,
        cd_multitask_solver_args,
    )

    with pytest.raises(ViolationError):
        cd_multitask_preprocess_data_kwargs("yes")  # type: ignore[arg-type]

    with pytest.raises(ViolationError):
        cd_multitask_fresh_coef_required(True, "present")  # type: ignore[arg-type]

    with pytest.raises(ViolationError):
        cd_multitask_initial_coef_zeros(0, 2, np.float64)

    with pytest.raises(ViolationError):
        cd_multitask_coef_fortran_array(np.array([[1.0, np.nan]]))

    with pytest.raises(ViolationError):
        cd_multitask_solver_args(object(), 1.0, 0.5, object(), object(), 0, 1e-4, object(), True)

    with pytest.raises(ViolationError):
        cd_multitask_solver_args(object(), 1.0, 0.5, object(), object(), 1, 1e-4, object(), "random")  # type: ignore[arg-type]
