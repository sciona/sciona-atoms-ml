from __future__ import annotations

import pytest
from icontract import ViolationError


def test_coordinate_descent_enet_path_prefit_grid_callback_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_enet_path_prefit_grid_callback_shell import (
        cd_enet_path_alpha_grid_18_result,
        cd_enet_path_prefit_18_result_unpack,
    )

    assert callable(cd_enet_path_prefit_18_result_unpack)
    assert callable(cd_enet_path_alpha_grid_18_result)


def test_coordinate_descent_enet_path_prefit_grid_callback_shell_matches_sklearn_assignments() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_enet_path_prefit_grid_callback_shell import (
        cd_enet_path_alpha_grid_18_result,
        cd_enet_path_prefit_18_result_unpack,
    )

    X = object()
    y = object()
    precompute = object()
    Xy = object()
    unpacked = cd_enet_path_prefit_18_result_unpack(
        (X, y, object(), object(), object(), precompute, Xy)
    )

    assert unpacked == {"X": X, "y": y, "precompute": precompute, "Xy": Xy}
    assert unpacked["X"] is X
    assert unpacked["y"] is y
    assert unpacked["precompute"] is precompute
    assert unpacked["Xy"] is Xy

    alphas = [3.0, 1.0]
    assert cd_enet_path_alpha_grid_18_result(alphas) is alphas


def test_coordinate_descent_enet_path_prefit_grid_callback_shell_contracts() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_enet_path_prefit_grid_callback_shell import (
        cd_enet_path_alpha_grid_18_result,
        cd_enet_path_prefit_18_result_unpack,
    )

    with pytest.raises(ViolationError):
        cd_enet_path_prefit_18_result_unpack((object(), object()))

    with pytest.raises(ViolationError):
        cd_enet_path_prefit_18_result_unpack("not-a-prefit-result")  # type: ignore[arg-type]

    with pytest.raises(ViolationError):
        cd_enet_path_alpha_grid_18_result(None)
