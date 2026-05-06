from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError


def test_coordinate_descent_cv_alpha_bookkeeping_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_cv_alpha_bookkeeping import (
        cd_cv_alpha_grid_required,
        cd_cv_default_l1_ratios,
        cd_cv_first_path_l1_ratio,
        cd_cv_has_l1_ratio_param,
        cd_cv_l1_ratios,
        cd_cv_n_alphas,
        cd_cv_n_l1_ratio,
        cd_cv_sorted_alphas,
    )

    assert callable(cd_cv_has_l1_ratio_param)
    assert callable(cd_cv_l1_ratios)
    assert callable(cd_cv_first_path_l1_ratio)
    assert callable(cd_cv_default_l1_ratios)
    assert callable(cd_cv_alpha_grid_required)
    assert callable(cd_cv_sorted_alphas)
    assert callable(cd_cv_n_l1_ratio)
    assert callable(cd_cv_n_alphas)


def test_coordinate_descent_cv_alpha_bookkeeping_matches_sklearn_shells() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_cv_alpha_bookkeeping import (
        cd_cv_alpha_grid_required,
        cd_cv_default_l1_ratios,
        cd_cv_first_path_l1_ratio,
        cd_cv_has_l1_ratio_param,
        cd_cv_l1_ratios,
        cd_cv_n_alphas,
        cd_cv_n_l1_ratio,
        cd_cv_sorted_alphas,
    )

    path_params = {"l1_ratio": [0.2, 0.8], "tol": 1e-4}
    assert cd_cv_has_l1_ratio_param(path_params) is True
    l1_ratios = cd_cv_l1_ratios(path_params["l1_ratio"])
    assert np.array_equal(l1_ratios, np.array([0.2, 0.8]))
    assert cd_cv_first_path_l1_ratio(l1_ratios) == pytest.approx(0.2)
    assert cd_cv_default_l1_ratios(False) == [1]
    assert cd_cv_alpha_grid_required(True) is True
    sorted_alphas = cd_cv_sorted_alphas(np.array([0.3, 0.1, 0.2]), 2)
    assert np.array_equal(sorted_alphas, np.array([[0.3, 0.2, 0.1], [0.3, 0.2, 0.1]]))
    assert cd_cv_n_l1_ratio(l1_ratios) == 2
    assert cd_cv_n_alphas(sorted_alphas) == 3


def test_coordinate_descent_cv_alpha_bookkeeping_contracts() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_cv_alpha_bookkeeping import (
        cd_cv_default_l1_ratios,
        cd_cv_sorted_alphas,
    )

    with pytest.raises(ViolationError):
        cd_cv_default_l1_ratios(True)

    with pytest.raises(ViolationError):
        cd_cv_sorted_alphas(np.array([]), 1)
