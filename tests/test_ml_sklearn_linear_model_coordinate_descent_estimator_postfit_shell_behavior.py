from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError


def test_coordinate_descent_estimator_postfit_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_estimator_postfit_shell import (
        cd_estimator_nonfinite_parameter_guard_required,
        cd_estimator_nonfinite_parameter_message,
        cd_estimator_single_target_branch,
        cd_estimator_single_target_coef,
        cd_estimator_single_target_dual_gap,
        cd_estimator_single_target_n_iter,
        cd_estimator_sparse_coef,
    )

    assert callable(cd_estimator_single_target_branch)
    assert callable(cd_estimator_single_target_n_iter)
    assert callable(cd_estimator_single_target_coef)
    assert callable(cd_estimator_single_target_dual_gap)
    assert callable(cd_estimator_nonfinite_parameter_guard_required)
    assert callable(cd_estimator_nonfinite_parameter_message)
    assert callable(cd_estimator_sparse_coef)


def test_coordinate_descent_estimator_postfit_shell_matches_sklearn_shells() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_estimator_postfit_shell import (
        cd_estimator_nonfinite_parameter_guard_required,
        cd_estimator_nonfinite_parameter_message,
        cd_estimator_single_target_branch,
        cd_estimator_single_target_coef,
        cd_estimator_single_target_dual_gap,
        cd_estimator_single_target_n_iter,
        cd_estimator_sparse_coef,
    )

    coef_matrix = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    dual_gaps = np.array([0.25, 0.5], dtype=np.float64)
    assert cd_estimator_single_target_branch(1) is True
    assert cd_estimator_single_target_n_iter([7, 8]) == 7
    assert np.array_equal(cd_estimator_single_target_coef(coef_matrix), np.array([1.0, 2.0]))
    assert np.isclose(cd_estimator_single_target_dual_gap(dual_gaps), 0.25)
    assert cd_estimator_nonfinite_parameter_guard_required(np.array([1.0]), 0.0) is False
    assert "non-finite parameter" in cd_estimator_nonfinite_parameter_message(np.array([1.0]), 0.0)
    sparse_coef = cd_estimator_sparse_coef(np.array([1.0, 0.0], dtype=np.float64))
    assert sparse_coef.shape == (1, 2)
    assert np.array_equal(sparse_coef.toarray(), np.array([[1.0, 0.0]], dtype=np.float64))


def test_coordinate_descent_estimator_postfit_shell_contracts() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_estimator_postfit_shell import (
        cd_estimator_single_target_branch,
        cd_estimator_single_target_n_iter,
    )

    with pytest.raises(ViolationError):
        cd_estimator_single_target_branch(0)

    with pytest.raises(ViolationError):
        cd_estimator_single_target_n_iter([])
