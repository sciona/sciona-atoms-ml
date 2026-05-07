from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError


def test_coordinate_descent_estimator_loop_tail_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_estimator_loop_tail_shell import (
        cd_estimator_coef_matrix_with_target,
        cd_estimator_dual_gaps_with_target,
        cd_estimator_n_iter_with_target,
        cd_estimator_target_coef_column,
        cd_estimator_target_dual_gap_scalar,
        cd_estimator_target_iteration_count,
    )

    assert callable(cd_estimator_target_coef_column)
    assert callable(cd_estimator_coef_matrix_with_target)
    assert callable(cd_estimator_target_dual_gap_scalar)
    assert callable(cd_estimator_dual_gaps_with_target)
    assert callable(cd_estimator_target_iteration_count)
    assert callable(cd_estimator_n_iter_with_target)


def test_coordinate_descent_estimator_loop_tail_shell_matches_sklearn_updates() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_estimator_loop_tail_shell import (
        cd_estimator_coef_matrix_with_target,
        cd_estimator_dual_gaps_with_target,
        cd_estimator_n_iter_with_target,
        cd_estimator_target_coef_column,
        cd_estimator_target_dual_gap_scalar,
        cd_estimator_target_iteration_count,
    )

    coef_matrix = np.zeros((2, 3), dtype=np.float64, order="F")
    this_coef = np.array([[1.0], [2.0], [3.0]], dtype=np.float64)

    target_coef = cd_estimator_target_coef_column(this_coef)
    assert np.array_equal(target_coef, this_coef[:, 0])

    updated_coef = cd_estimator_coef_matrix_with_target(coef_matrix, 1, this_coef)
    assert updated_coef is not coef_matrix
    assert np.array_equal(updated_coef[0], coef_matrix[0])
    assert np.array_equal(updated_coef[1], np.array([1.0, 2.0, 3.0]))
    assert np.count_nonzero(coef_matrix) == 0

    dual_gaps = np.zeros(2, dtype=np.float32)
    this_dual_gap = np.array([0.25], dtype=np.float64)
    assert np.isclose(cd_estimator_target_dual_gap_scalar(this_dual_gap), 0.25)
    updated_dual_gaps = cd_estimator_dual_gaps_with_target(dual_gaps, 0, this_dual_gap)
    assert updated_dual_gaps is not dual_gaps
    assert updated_dual_gaps.dtype == np.float32
    assert np.isclose(updated_dual_gaps[0], 0.25)
    assert np.isclose(updated_dual_gaps[1], 0.0)

    this_iter = np.array([7], dtype=np.int64)
    assert cd_estimator_target_iteration_count(this_iter) == 7
    assert cd_estimator_n_iter_with_target([3, 5], this_iter) == [3, 5, 7]


def test_coordinate_descent_estimator_loop_tail_shell_contracts() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_estimator_loop_tail_shell import (
        cd_estimator_coef_matrix_with_target,
        cd_estimator_dual_gaps_with_target,
        cd_estimator_n_iter_with_target,
        cd_estimator_target_coef_column,
        cd_estimator_target_dual_gap_scalar,
        cd_estimator_target_iteration_count,
    )

    with pytest.raises(ViolationError):
        cd_estimator_target_coef_column(np.array([1.0, 2.0]))

    with pytest.raises(ViolationError):
        cd_estimator_coef_matrix_with_target(np.zeros((1, 2)), 0, np.ones((3, 1)))

    with pytest.raises(ViolationError):
        cd_estimator_target_dual_gap_scalar(np.array([np.inf]))

    with pytest.raises(ViolationError):
        cd_estimator_dual_gaps_with_target(np.zeros(1), 2, np.array([0.1]))

    with pytest.raises(ViolationError):
        cd_estimator_target_iteration_count(np.array([0]))

    with pytest.raises(ViolationError):
        cd_estimator_n_iter_with_target([2, 0], np.array([3]))
