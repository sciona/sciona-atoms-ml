from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError


def test_coordinate_descent_estimator_multitarget_postfit_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_estimator_multitarget_postfit_shell import (
        cd_estimator_multitarget_branch,
        cd_estimator_multitarget_coef,
        cd_estimator_multitarget_dual_gap,
    )

    assert callable(cd_estimator_multitarget_branch)
    assert callable(cd_estimator_multitarget_coef)
    assert callable(cd_estimator_multitarget_dual_gap)


def test_coordinate_descent_estimator_multitarget_postfit_shell_matches_assignments() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_estimator_multitarget_postfit_shell import (
        cd_estimator_multitarget_branch,
        cd_estimator_multitarget_coef,
        cd_estimator_multitarget_dual_gap,
    )

    coef = np.array([[1.0, 0.0], [0.5, -0.25]], dtype=np.float64)
    dual_gaps = np.array([0.01, 0.02], dtype=np.float64)

    assert cd_estimator_multitarget_branch(2) is True
    assert cd_estimator_multitarget_branch(3) is True
    assert cd_estimator_multitarget_branch(1) is False
    assert cd_estimator_multitarget_coef(coef, 2) is coef
    assert cd_estimator_multitarget_dual_gap(dual_gaps, 2) is dual_gaps


def test_coordinate_descent_estimator_multitarget_postfit_shell_contracts() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_estimator_multitarget_postfit_shell import (
        cd_estimator_multitarget_coef,
        cd_estimator_multitarget_dual_gap,
    )

    with pytest.raises(ViolationError):
        cd_estimator_multitarget_coef(np.array([[1.0, 2.0]]), 1)

    with pytest.raises(ViolationError):
        cd_estimator_multitarget_dual_gap(np.array([0.1]), 1)

    with pytest.raises(ViolationError):
        cd_estimator_multitarget_coef(np.array([[1.0], [2.0]]), 3)

    with pytest.raises(ViolationError):
        cd_estimator_multitarget_dual_gap(np.array([0.1, 0.2]), 3)
