from __future__ import annotations

import pytest
from icontract import ViolationError


def test_coordinate_descent_multitask_solver_result_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_multitask_solver_result_shell import (
        cd_multitask_set_intercept_args,
        cd_multitask_solver_result_coef,
        cd_multitask_solver_result_dual_gap,
        cd_multitask_solver_result_eps,
        cd_multitask_solver_result_n_iter,
    )

    assert callable(cd_multitask_solver_result_coef)
    assert callable(cd_multitask_solver_result_dual_gap)
    assert callable(cd_multitask_solver_result_eps)
    assert callable(cd_multitask_solver_result_n_iter)
    assert callable(cd_multitask_set_intercept_args)


def test_coordinate_descent_multitask_solver_result_shell_matches_sklearn_tail() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_multitask_solver_result_shell import (
        cd_multitask_set_intercept_args,
        cd_multitask_solver_result_coef,
        cd_multitask_solver_result_dual_gap,
        cd_multitask_solver_result_eps,
        cd_multitask_solver_result_n_iter,
    )

    coef = object()
    dual_gap = object()
    eps = object()
    n_iter = object()
    solver_result = (coef, dual_gap, eps, n_iter)

    assert cd_multitask_solver_result_coef(solver_result) is coef
    assert cd_multitask_solver_result_dual_gap(solver_result) is dual_gap
    assert cd_multitask_solver_result_eps(solver_result) is eps
    assert cd_multitask_solver_result_n_iter(solver_result) is n_iter

    X_offset = object()
    y_offset = object()
    X_scale = object()
    args = cd_multitask_set_intercept_args(X_offset, y_offset, X_scale)
    assert args == (X_offset, y_offset, X_scale)
    assert args[0] is X_offset
    assert args[1] is y_offset
    assert args[2] is X_scale


def test_coordinate_descent_multitask_solver_result_shell_contracts() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_multitask_solver_result_shell import (
        cd_multitask_set_intercept_args,
        cd_multitask_solver_result_coef,
        cd_multitask_solver_result_dual_gap,
        cd_multitask_solver_result_eps,
        cd_multitask_solver_result_n_iter,
    )

    with pytest.raises(ViolationError):
        cd_multitask_solver_result_coef((object(), object(), object()))

    with pytest.raises(ViolationError):
        cd_multitask_solver_result_dual_gap((object(), object(), object()))

    with pytest.raises(ViolationError):
        cd_multitask_solver_result_eps((object(), object(), object()))

    with pytest.raises(ViolationError):
        cd_multitask_solver_result_n_iter((object(), object(), object()))

    with pytest.raises(ViolationError):
        cd_multitask_set_intercept_args(None, object(), object())
