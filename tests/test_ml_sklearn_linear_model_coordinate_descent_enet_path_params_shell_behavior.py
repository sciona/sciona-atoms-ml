from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError


def test_coordinate_descent_enet_path_params_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_enet_path_params_shell import (
        cd_enet_path_popped_params,
    )

    assert callable(cd_enet_path_popped_params)


def test_coordinate_descent_enet_path_params_shell_matches_defaults() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_enet_path_params_shell import (
        cd_enet_path_popped_params,
    )

    result = cd_enet_path_popped_params({})

    assert result == {
        "X_offset_param": None,
        "X_scale_param": None,
        "sample_weight": None,
        "tol": 1e-4,
        "max_iter": 1000,
        "random_state": None,
        "selection": "cyclic",
        "remaining_params": {},
    }


def test_coordinate_descent_enet_path_params_shell_matches_explicit_pops() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_enet_path_params_shell import (
        cd_enet_path_popped_params,
    )

    X_offset = np.array([1.0, 2.0], dtype=np.float64)
    X_scale = np.array([3.0, 4.0], dtype=np.float64)
    sample_weight = np.array([0.25, 0.75], dtype=np.float64)
    random_state = object()
    params = {
        "X_offset": X_offset,
        "X_scale": X_scale,
        "sample_weight": sample_weight,
        "tol": 1e-5,
        "max_iter": 12,
        "random_state": random_state,
        "selection": "random",
        "unexpected": "leftover",
    }

    result = cd_enet_path_popped_params(params)

    assert result["X_offset_param"] is X_offset
    assert result["X_scale_param"] is X_scale
    assert result["sample_weight"] is sample_weight
    assert result["tol"] == 1e-5
    assert result["max_iter"] == 12
    assert result["random_state"] is random_state
    assert result["selection"] == "random"
    assert result["remaining_params"] == {"unexpected": "leftover"}
    assert set(params) == {
        "X_offset",
        "X_scale",
        "sample_weight",
        "tol",
        "max_iter",
        "random_state",
        "selection",
        "unexpected",
    }


def test_coordinate_descent_enet_path_params_shell_contracts() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_enet_path_params_shell import (
        cd_enet_path_popped_params,
    )

    with pytest.raises(ViolationError):
        cd_enet_path_popped_params([("tol", 1e-4)])  # type: ignore[arg-type]

    with pytest.raises(ViolationError):
        cd_enet_path_popped_params({"max_iter": 0})

    with pytest.raises(ViolationError):
        cd_enet_path_popped_params({"selection": object()})
