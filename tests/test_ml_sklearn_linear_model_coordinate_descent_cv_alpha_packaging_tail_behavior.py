from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError


def test_coordinate_descent_cv_alpha_packaging_tail_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_cv_alpha_packaging_tail import (
        cd_cv_auto_alphas_array,
        cd_cv_auto_alphas_packaging_required,
        cd_cv_auto_alphas_public,
        cd_cv_auto_alphas_single_ratio_collapse_required,
        cd_cv_user_alphas_packaging_required,
        cd_cv_user_alphas_public,
    )

    assert callable(cd_cv_auto_alphas_packaging_required)
    assert callable(cd_cv_user_alphas_packaging_required)
    assert callable(cd_cv_auto_alphas_array)
    assert callable(cd_cv_auto_alphas_single_ratio_collapse_required)
    assert callable(cd_cv_auto_alphas_public)
    assert callable(cd_cv_user_alphas_public)


def test_coordinate_descent_cv_alpha_packaging_tail_matches_sklearn_shells() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_cv_alpha_packaging_tail import (
        cd_cv_auto_alphas_array,
        cd_cv_auto_alphas_packaging_required,
        cd_cv_auto_alphas_public,
        cd_cv_auto_alphas_single_ratio_collapse_required,
        cd_cv_user_alphas_packaging_required,
        cd_cv_user_alphas_public,
    )

    assert cd_cv_auto_alphas_packaging_required(True) is True
    assert cd_cv_user_alphas_packaging_required(True) is False

    alphas = [[3.0, 2.0, 1.0]]
    auto_array = cd_cv_auto_alphas_array(alphas, True)
    assert np.array_equal(auto_array, np.array([[3.0, 2.0, 1.0]]))
    collapse_required = cd_cv_auto_alphas_single_ratio_collapse_required(1)
    assert collapse_required is True
    assert np.array_equal(
        cd_cv_auto_alphas_public(auto_array, collapse_required),
        np.array([3.0, 2.0, 1.0]),
    )

    multi_array = cd_cv_auto_alphas_array([[4.0, 2.0], [3.0, 1.0]], True)
    assert cd_cv_auto_alphas_single_ratio_collapse_required(2) is False
    assert np.array_equal(cd_cv_auto_alphas_public(multi_array, False), multi_array)

    user_alphas = np.array([[5.0, 2.0, 1.0], [5.0, 2.0, 1.0]])
    assert cd_cv_user_alphas_packaging_required(False) is True
    assert np.array_equal(
        cd_cv_user_alphas_public(user_alphas, True),
        np.array([5.0, 2.0, 1.0]),
    )


def test_coordinate_descent_cv_alpha_packaging_tail_contracts() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_cv_alpha_packaging_tail import (
        cd_cv_auto_alphas_array,
        cd_cv_auto_alphas_packaging_required,
        cd_cv_auto_alphas_public,
        cd_cv_auto_alphas_single_ratio_collapse_required,
        cd_cv_user_alphas_public,
    )

    with pytest.raises(ViolationError):
        cd_cv_auto_alphas_packaging_required("yes")  # type: ignore[arg-type]

    with pytest.raises(ViolationError):
        cd_cv_auto_alphas_array([[1.0]], False)

    with pytest.raises(ViolationError):
        cd_cv_auto_alphas_single_ratio_collapse_required(0)

    with pytest.raises(ViolationError):
        cd_cv_auto_alphas_public(np.array([]), False)

    with pytest.raises(ViolationError):
        cd_cv_user_alphas_public([[1.0]], False)
