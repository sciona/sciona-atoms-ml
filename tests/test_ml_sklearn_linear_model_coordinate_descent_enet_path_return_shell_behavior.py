from __future__ import annotations

import pytest
from icontract import ViolationError


def test_coordinate_descent_enet_path_return_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_enet_path_return_shell import (
        cd_enet_path_result_tuple,
        cd_enet_path_return_arity,
    )

    assert callable(cd_enet_path_return_arity)
    assert callable(cd_enet_path_result_tuple)


def test_coordinate_descent_enet_path_return_shell_matches_sklearn_branch() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_enet_path_return_shell import (
        cd_enet_path_result_tuple,
        cd_enet_path_return_arity,
    )

    alphas = object()
    coefs = object()
    dual_gaps = object()
    n_iters = [5, 7, 11]

    assert cd_enet_path_return_arity(False) == 3
    assert cd_enet_path_return_arity(True) == 4

    without_iters = cd_enet_path_result_tuple(alphas, coefs, dual_gaps, n_iters, False)
    assert without_iters == (alphas, coefs, dual_gaps)
    assert len(without_iters) == 3

    with_iters = cd_enet_path_result_tuple(alphas, coefs, dual_gaps, n_iters, True)
    assert with_iters == (alphas, coefs, dual_gaps, n_iters)
    assert with_iters[3] is n_iters


def test_coordinate_descent_enet_path_return_shell_contracts() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_enet_path_return_shell import (
        cd_enet_path_result_tuple,
        cd_enet_path_return_arity,
    )

    with pytest.raises(ViolationError):
        cd_enet_path_return_arity(1)  # type: ignore[arg-type]

    with pytest.raises(ViolationError):
        cd_enet_path_result_tuple("alphas", "coefs", "dual_gaps", "not-seq", False)
