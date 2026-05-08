from __future__ import annotations

import pytest
from icontract import ViolationError


def test_coordinate_descent_enet_path_solver_payload_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_enet_path_solver_payload_shell import (
        cd_enet_path_dense_solver_args,
        cd_enet_path_gram_solver_args,
        cd_enet_path_multitask_solver_args,
        cd_enet_path_sparse_solver_kwargs,
    )

    assert callable(cd_enet_path_sparse_solver_kwargs)
    assert callable(cd_enet_path_multitask_solver_args)
    assert callable(cd_enet_path_gram_solver_args)
    assert callable(cd_enet_path_dense_solver_args)


def test_coordinate_descent_enet_path_solver_payload_shell_matches_sklearn_calls() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_enet_path_solver_payload_shell import (
        cd_enet_path_dense_solver_args,
        cd_enet_path_gram_solver_args,
        cd_enet_path_multitask_solver_args,
        cd_enet_path_sparse_solver_kwargs,
    )

    coef = object()
    l1_reg = object()
    l2_reg = object()
    X = object()
    X_data = object()
    X_indices = object()
    X_indptr = object()
    X_sparse_scaling = object()
    Xy = object()
    y = object()
    sample_weight = object()
    precompute = object()
    tol = object()
    rng = object()

    sparse_kwargs = cd_enet_path_sparse_solver_kwargs(
        coef,
        l1_reg,
        l2_reg,
        X_data,
        X_indices,
        X_indptr,
        y,
        sample_weight,
        X_sparse_scaling,
        1000,
        tol,
        rng,
        True,
        False,
    )
    assert sparse_kwargs == {
        "w": coef,
        "alpha": l1_reg,
        "beta": l2_reg,
        "X_data": X_data,
        "X_indices": X_indices,
        "X_indptr": X_indptr,
        "y": y,
        "sample_weight": sample_weight,
        "X_mean": X_sparse_scaling,
        "max_iter": 1000,
        "tol": tol,
        "rng": rng,
        "random": True,
        "positive": False,
    }
    assert sparse_kwargs["w"] is coef
    assert sparse_kwargs["X_data"] is X_data
    assert sparse_kwargs["sample_weight"] is sample_weight
    assert sparse_kwargs["X_mean"] is X_sparse_scaling

    multitask_args = cd_enet_path_multitask_solver_args(
        coef, l1_reg, l2_reg, X, y, 1000, tol, rng, True
    )
    assert multitask_args == (coef, l1_reg, l2_reg, X, y, 1000, tol, rng, True)
    assert multitask_args[0] is coef
    assert multitask_args[3] is X
    assert multitask_args[7] is rng

    gram_args = cd_enet_path_gram_solver_args(
        coef, l1_reg, l2_reg, precompute, Xy, y, 500, tol, rng, False, True
    )
    assert gram_args == (
        coef,
        l1_reg,
        l2_reg,
        precompute,
        Xy,
        y,
        500,
        tol,
        rng,
        False,
        True,
    )
    assert gram_args[3] is precompute
    assert gram_args[4] is Xy
    assert gram_args[8] is rng

    dense_args = cd_enet_path_dense_solver_args(
        coef, l1_reg, l2_reg, X, y, 250, tol, rng, False, False
    )
    assert dense_args == (coef, l1_reg, l2_reg, X, y, 250, tol, rng, False, False)
    assert dense_args[0] is coef
    assert dense_args[3] is X
    assert dense_args[7] is rng


def test_coordinate_descent_enet_path_solver_payload_shell_contracts() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_enet_path_solver_payload_shell import (
        cd_enet_path_dense_solver_args,
        cd_enet_path_gram_solver_args,
        cd_enet_path_multitask_solver_args,
        cd_enet_path_sparse_solver_kwargs,
    )

    with pytest.raises(ViolationError):
        cd_enet_path_sparse_solver_kwargs(
            object(),
            object(),
            object(),
            object(),
            object(),
            object(),
            object(),
            None,
            object(),
            0,
            object(),
            object(),
            True,
            False,
        )

    with pytest.raises(ViolationError):
        cd_enet_path_multitask_solver_args(
            object(), object(), object(), object(), object(), 1, object(), object(), "yes"
        )  # type: ignore[arg-type]

    with pytest.raises(ViolationError):
        cd_enet_path_gram_solver_args(
            object(),
            object(),
            object(),
            object(),
            object(),
            object(),
            1,
            object(),
            object(),
            False,
            "yes",
        )  # type: ignore[arg-type]

    with pytest.raises(ViolationError):
        cd_enet_path_dense_solver_args(
            object(), object(), object(), object(), object(), 0, object(), object(), False, False
        )
