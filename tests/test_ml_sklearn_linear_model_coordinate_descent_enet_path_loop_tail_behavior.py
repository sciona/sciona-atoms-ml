from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError


def test_coordinate_descent_enet_path_loop_tail_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_enet_path_loop_tail import (
        cd_enet_path_model_coef,
        cd_enet_path_model_iteration_count,
        cd_enet_path_scaled_dual_gap,
        cd_enet_path_selection_error_message,
        cd_enet_path_selection_guard_required,
        cd_enet_path_verbose_progress_message,
        cd_enet_path_verbose_use_progress_print,
        cd_enet_path_verbose_use_stderr_dot,
        cd_enet_path_verbose_use_tuple_print,
    )

    assert callable(cd_enet_path_selection_guard_required)
    assert callable(cd_enet_path_selection_error_message)
    assert callable(cd_enet_path_model_coef)
    assert callable(cd_enet_path_scaled_dual_gap)
    assert callable(cd_enet_path_model_iteration_count)
    assert callable(cd_enet_path_verbose_use_tuple_print)
    assert callable(cd_enet_path_verbose_use_progress_print)
    assert callable(cd_enet_path_verbose_use_stderr_dot)
    assert callable(cd_enet_path_verbose_progress_message)


def test_coordinate_descent_enet_path_loop_tail_matches_sklearn_shells() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_enet_path_loop_tail import (
        cd_enet_path_model_coef,
        cd_enet_path_model_iteration_count,
        cd_enet_path_scaled_dual_gap,
        cd_enet_path_selection_error_message,
        cd_enet_path_selection_guard_required,
        cd_enet_path_verbose_progress_message,
        cd_enet_path_verbose_use_progress_print,
        cd_enet_path_verbose_use_stderr_dot,
        cd_enet_path_verbose_use_tuple_print,
    )

    model = (np.array([1.0, 2.0], dtype=np.float64), 0.6, 0.0, 7)
    assert cd_enet_path_selection_guard_required("bad") is True
    assert cd_enet_path_selection_error_message("bad") == "selection should be either random or cyclic."
    assert np.array_equal(cd_enet_path_model_coef(model), np.array([1.0, 2.0], dtype=np.float64))
    assert np.isclose(cd_enet_path_scaled_dual_gap(0.6, 3), 0.2)
    assert cd_enet_path_model_iteration_count(model) == 7
    assert cd_enet_path_verbose_use_tuple_print(3) is True
    assert cd_enet_path_verbose_use_progress_print(2) is True
    assert cd_enet_path_verbose_use_stderr_dot(1) is True
    assert cd_enet_path_verbose_progress_message(4, 12) == "Path: 004 out of 012"


def test_coordinate_descent_enet_path_loop_tail_contracts() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_enet_path_loop_tail import (
        cd_enet_path_model_iteration_count,
        cd_enet_path_scaled_dual_gap,
    )

    with pytest.raises(ViolationError):
        cd_enet_path_scaled_dual_gap(1.0, 0)

    with pytest.raises(ViolationError):
        cd_enet_path_model_iteration_count((np.array([1.0]), 0.1, 0.0))
