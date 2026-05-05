from __future__ import annotations

import pytest
from icontract import ViolationError


def test_coordinate_descent_enet_path_solver_dispatch_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_enet_path_solver_dispatch import (
        cd_enet_path_gram_validation_required,
        cd_enet_path_invalid_precompute_message,
        cd_enet_path_use_dense_solver,
        cd_enet_path_use_gram_solver,
        cd_enet_path_use_multi_task_solver,
        cd_enet_path_use_sparse_solver,
    )

    assert callable(cd_enet_path_gram_validation_required)
    assert callable(cd_enet_path_use_sparse_solver)
    assert callable(cd_enet_path_use_multi_task_solver)
    assert callable(cd_enet_path_use_gram_solver)
    assert callable(cd_enet_path_use_dense_solver)
    assert callable(cd_enet_path_invalid_precompute_message)


def test_coordinate_descent_enet_path_solver_dispatch_matches_sklearn_shells() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_enet_path_solver_dispatch import (
        cd_enet_path_gram_validation_required,
        cd_enet_path_invalid_precompute_message,
        cd_enet_path_use_dense_solver,
        cd_enet_path_use_gram_solver,
        cd_enet_path_use_multi_task_solver,
        cd_enet_path_use_sparse_solver,
    )

    assert cd_enet_path_gram_validation_required(True, True) is True
    assert cd_enet_path_use_sparse_solver(multi_output=False, x_is_sparse=True) is True
    assert cd_enet_path_use_multi_task_solver(use_sparse_solver=False, multi_output=True) is True
    assert cd_enet_path_use_gram_solver(
        use_sparse_solver=False, multi_output=False, precompute_is_array=True
    ) is True
    assert cd_enet_path_use_dense_solver(
        use_sparse_solver=False, use_multi_task_solver=False, precompute_is_false=True
    ) is True
    assert (
        cd_enet_path_invalid_precompute_message("bad")
        == "Precompute should be one of True, False, 'auto' or array-like. Got 'bad'"
    )


def test_coordinate_descent_enet_path_solver_dispatch_contracts() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_enet_path_solver_dispatch import (
        cd_enet_path_gram_validation_required,
        cd_enet_path_use_dense_solver,
    )

    with pytest.raises(ViolationError):
        cd_enet_path_gram_validation_required(1, True)

    with pytest.raises(ViolationError):
        cd_enet_path_use_dense_solver(False, False, "no")
