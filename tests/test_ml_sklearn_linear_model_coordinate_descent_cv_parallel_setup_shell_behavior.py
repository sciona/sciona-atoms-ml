from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError


def test_coordinate_descent_cv_parallel_setup_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_cv_parallel_setup_shell import (
        cd_cv_best_mse_initial,
        cd_cv_fold_count,
        cd_cv_folds,
        cd_cv_path_job_count,
        cd_cv_path_job_kwargs,
    )

    assert callable(cd_cv_folds)
    assert callable(cd_cv_fold_count)
    assert callable(cd_cv_path_job_kwargs)
    assert callable(cd_cv_path_job_count)
    assert callable(cd_cv_best_mse_initial)


def test_coordinate_descent_cv_parallel_setup_shell_matches_sklearn_shells() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_cv_parallel_setup_shell import (
        cd_cv_best_mse_initial,
        cd_cv_fold_count,
        cd_cv_folds,
        cd_cv_path_job_count,
        cd_cv_path_job_kwargs,
    )

    splits = [(np.array([0, 1]), np.array([2])), (np.array([2]), np.array([0, 1]))]
    folds = cd_cv_folds(splits)
    assert folds == splits
    assert cd_cv_fold_count(folds) == 2
    job_kwargs = cd_cv_path_job_kwargs(np.array([0.5, 0.1]), 0.7, np.float64)
    assert np.array_equal(job_kwargs["alphas"], np.array([0.5, 0.1]))
    assert job_kwargs["l1_ratio"] == 0.7
    assert job_kwargs["X_order"] == "F"
    assert job_kwargs["dtype"] is np.float64
    assert cd_cv_path_job_count([0.2, 0.8], folds) == 4
    assert np.isinf(cd_cv_best_mse_initial(2))


def test_coordinate_descent_cv_parallel_setup_shell_contracts() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_cv_parallel_setup_shell import (
        cd_cv_best_mse_initial,
        cd_cv_folds,
        cd_cv_path_job_count,
    )

    with pytest.raises(ViolationError):
        cd_cv_folds([np.array([0, 1])])

    with pytest.raises(ViolationError):
        cd_cv_path_job_count(object(), [(1, 2)])

    with pytest.raises(ViolationError):
        cd_cv_best_mse_initial(0)
