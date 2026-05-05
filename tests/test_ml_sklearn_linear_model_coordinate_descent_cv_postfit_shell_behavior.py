from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError


def test_coordinate_descent_cv_postfit_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_cv_postfit_shell import (
        cd_cv_delete_l1_ratio_required,
        cd_cv_fit_coef,
        cd_cv_fit_dual_gap,
        cd_cv_fit_intercept,
        cd_cv_fit_n_iter,
        cd_cv_fit_return_self,
        cd_cv_refit_uses_sample_weight,
    )

    assert callable(cd_cv_refit_uses_sample_weight)
    assert callable(cd_cv_delete_l1_ratio_required)
    assert callable(cd_cv_fit_coef)
    assert callable(cd_cv_fit_intercept)
    assert callable(cd_cv_fit_dual_gap)
    assert callable(cd_cv_fit_n_iter)
    assert callable(cd_cv_fit_return_self)


def test_coordinate_descent_cv_postfit_shell_matches_sklearn_shells() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_cv_postfit_shell import (
        cd_cv_delete_l1_ratio_required,
        cd_cv_fit_coef,
        cd_cv_fit_dual_gap,
        cd_cv_fit_intercept,
        cd_cv_fit_n_iter,
        cd_cv_fit_return_self,
        cd_cv_refit_uses_sample_weight,
    )

    assert cd_cv_refit_uses_sample_weight(np.array([1.0, 2.0], dtype=np.float64)) is True
    assert cd_cv_delete_l1_ratio_required(False) is True
    assert np.array_equal(cd_cv_fit_coef(np.array([1.0, 2.0], dtype=np.float64)), np.array([1.0, 2.0]))
    assert np.isclose(cd_cv_fit_intercept(0.5), 0.5)
    assert np.isclose(cd_cv_fit_dual_gap(0.1), 0.1)
    assert cd_cv_fit_n_iter([3, 4]) == [3, 4]
    token = object()
    assert cd_cv_fit_return_self(token) is token


def test_coordinate_descent_cv_postfit_shell_contracts() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_cv_postfit_shell import (
        cd_cv_delete_l1_ratio_required,
        cd_cv_fit_n_iter,
    )

    with pytest.raises(ViolationError):
        cd_cv_delete_l1_ratio_required(1)

    with pytest.raises(ViolationError):
        cd_cv_fit_n_iter([])
