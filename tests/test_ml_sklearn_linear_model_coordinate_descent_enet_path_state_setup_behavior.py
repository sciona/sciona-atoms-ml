from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError


def test_coordinate_descent_enet_path_state_setup_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_enet_path_state_setup import (
        cd_enet_path_alpha_count,
        cd_enet_path_coef_path_buffer,
        cd_enet_path_coef_path_shape,
        cd_enet_path_dual_gap_buffer,
        cd_enet_path_initial_coef,
        cd_enet_path_initial_coef_required,
        cd_enet_path_iteration_buffer,
    )

    assert callable(cd_enet_path_alpha_count)
    assert callable(cd_enet_path_dual_gap_buffer)
    assert callable(cd_enet_path_iteration_buffer)
    assert callable(cd_enet_path_coef_path_shape)
    assert callable(cd_enet_path_coef_path_buffer)
    assert callable(cd_enet_path_initial_coef_required)
    assert callable(cd_enet_path_initial_coef)


def test_coordinate_descent_enet_path_state_setup_matches_sklearn_shells() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_enet_path_state_setup import (
        cd_enet_path_alpha_count,
        cd_enet_path_coef_path_buffer,
        cd_enet_path_coef_path_shape,
        cd_enet_path_dual_gap_buffer,
        cd_enet_path_initial_coef,
        cd_enet_path_initial_coef_required,
        cd_enet_path_iteration_buffer,
    )

    alpha_count = cd_enet_path_alpha_count([1.0, 0.5, 0.1])
    assert alpha_count == 3
    assert cd_enet_path_dual_gap_buffer(alpha_count).shape == (3,)
    assert cd_enet_path_iteration_buffer(alpha_count) == []
    assert cd_enet_path_coef_path_shape(4, 3, False, None) == (4, 3)
    assert cd_enet_path_coef_path_shape(4, 3, True, 2) == (2, 4, 3)
    assert cd_enet_path_coef_path_buffer((4, 3), "float32").dtype == np.float32
    assert cd_enet_path_initial_coef_required(None) is True
    zeros = cd_enet_path_initial_coef((4, 3), "float64", None)
    assert zeros.shape == (4,)
    assert np.allclose(zeros, np.zeros(4, dtype=np.float64))
    warm = cd_enet_path_initial_coef((2, 4, 3), "float64", np.ones((2, 4), dtype=np.float64))
    assert warm.shape == (2, 4)
    assert warm.flags["F_CONTIGUOUS"]


def test_coordinate_descent_enet_path_state_setup_contracts() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_enet_path_state_setup import (
        cd_enet_path_alpha_count,
        cd_enet_path_initial_coef,
    )

    with pytest.raises(ViolationError):
        cd_enet_path_alpha_count([])

    with pytest.raises(ViolationError):
        cd_enet_path_initial_coef((4, 3), "float64", np.ones((2,), dtype=np.float64))
