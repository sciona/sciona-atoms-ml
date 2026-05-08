from __future__ import annotations

import pytest
from icontract import ViolationError


def test_coordinate_descent_enet_path_screening_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_enet_path_screening_shell import (
        cd_enet_path_dense_screening_args,
        cd_enet_path_do_screening_param,
        cd_enet_path_gram_screening_args,
        cd_enet_path_multitask_screening_args,
        cd_enet_path_sparse_screening_kwarg,
    )

    assert callable(cd_enet_path_do_screening_param)
    assert callable(cd_enet_path_sparse_screening_kwarg)
    assert callable(cd_enet_path_multitask_screening_args)
    assert callable(cd_enet_path_gram_screening_args)
    assert callable(cd_enet_path_dense_screening_args)


def test_coordinate_descent_enet_path_screening_shell_matches_sklearn_delta() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_enet_path_screening_shell import (
        cd_enet_path_dense_screening_args,
        cd_enet_path_do_screening_param,
        cd_enet_path_gram_screening_args,
        cd_enet_path_multitask_screening_args,
        cd_enet_path_sparse_screening_kwarg,
    )

    assert cd_enet_path_do_screening_param({"tol": 1e-3}) == {
        "do_screening": True,
        "remaining_params": {"tol": 1e-3},
    }
    assert cd_enet_path_do_screening_param({"do_screening": False, "selection": "random"}) == {
        "do_screening": False,
        "remaining_params": {"selection": "random"},
    }

    assert cd_enet_path_sparse_screening_kwarg(False) == {"do_screening": False}
    assert cd_enet_path_multitask_screening_args(("coef", "l1", "l2"), True) == (
        "coef",
        "l1",
        "l2",
        True,
    )
    assert cd_enet_path_gram_screening_args(("coef", "gram"), False) == (
        "coef",
        "gram",
        False,
    )
    assert cd_enet_path_dense_screening_args(("coef", "X", "y"), True) == (
        "coef",
        "X",
        "y",
        True,
    )


def test_coordinate_descent_enet_path_screening_shell_contracts() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_enet_path_screening_shell import (
        cd_enet_path_dense_screening_args,
        cd_enet_path_do_screening_param,
        cd_enet_path_sparse_screening_kwarg,
    )

    with pytest.raises(ViolationError):
        cd_enet_path_do_screening_param({"do_screening": 1})

    with pytest.raises(ViolationError):
        cd_enet_path_sparse_screening_kwarg("yes")  # type: ignore[arg-type]

    with pytest.raises(ViolationError):
        cd_enet_path_dense_screening_args("not-a-sequence", True)  # type: ignore[arg-type]
