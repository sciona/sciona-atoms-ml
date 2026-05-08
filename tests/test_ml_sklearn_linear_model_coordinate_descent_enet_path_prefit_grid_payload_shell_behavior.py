from __future__ import annotations

import pytest
from icontract import ViolationError


def test_coordinate_descent_enet_path_prefit_grid_payload_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_enet_path_prefit_grid_payload_shell import (
        cd_enet_path_alpha_grid_18_kwargs,
        cd_enet_path_prefit_18_kwargs,
    )

    assert callable(cd_enet_path_prefit_18_kwargs)
    assert callable(cd_enet_path_alpha_grid_18_kwargs)


def test_coordinate_descent_enet_path_prefit_grid_payload_shell_matches_sklearn_payloads() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_enet_path_prefit_grid_payload_shell import (
        cd_enet_path_alpha_grid_18_kwargs,
        cd_enet_path_prefit_18_kwargs,
    )

    assert cd_enet_path_prefit_18_kwargs("enet_path") == {
        "fit_intercept": False,
        "copy": False,
        "check_gram": True,
    }

    xy = object()
    eps = 1e-3
    n_alphas = 100
    payload = cd_enet_path_alpha_grid_18_kwargs(
        "enet_path",
        Xy=xy,
        l1_ratio=0.5,
        eps=eps,
        n_alphas=n_alphas,
    )

    assert payload == {
        "Xy": xy,
        "l1_ratio": 0.5,
        "fit_intercept": False,
        "eps": eps,
        "n_alphas": n_alphas,
    }
    assert "copy_X" not in payload
    assert "sample_weight" not in payload


def test_coordinate_descent_enet_path_prefit_grid_payload_shell_contracts() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_enet_path_prefit_grid_payload_shell import (
        cd_enet_path_alpha_grid_18_kwargs,
        cd_enet_path_prefit_18_kwargs,
    )

    with pytest.raises(ViolationError):
        cd_enet_path_prefit_18_kwargs("lasso_path")

    with pytest.raises(ViolationError):
        cd_enet_path_alpha_grid_18_kwargs("lasso_path", None, 0.5, 1e-3, 100)

    with pytest.raises(ViolationError):
        cd_enet_path_alpha_grid_18_kwargs("enet_path", None, 0.5, 0.0, 100)

    with pytest.raises(ViolationError):
        cd_enet_path_alpha_grid_18_kwargs("enet_path", None, 0.5, 1e-3, 0)
