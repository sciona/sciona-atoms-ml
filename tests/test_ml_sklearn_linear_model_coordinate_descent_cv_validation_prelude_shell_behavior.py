from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError


def test_coordinate_descent_cv_validation_prelude_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_cv_validation_prelude_shell import (
        cd_cv_check_y_params,
        cd_cv_fit_params_guard_args,
        cd_cv_fortran_check_x_params,
        cd_cv_initial_copy_x,
        cd_cv_non_reference_copy_x,
        cd_cv_reference_check_x_params,
        cd_cv_reference_validation_copy_x,
    )

    assert callable(cd_cv_fit_params_guard_args)
    assert callable(cd_cv_initial_copy_x)
    assert callable(cd_cv_check_y_params)
    assert callable(cd_cv_reference_check_x_params)
    assert callable(cd_cv_fortran_check_x_params)
    assert callable(cd_cv_reference_validation_copy_x)
    assert callable(cd_cv_non_reference_copy_x)


def test_coordinate_descent_cv_validation_prelude_shell_matches_sklearn_shells() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_cv_validation_prelude_shell import (
        cd_cv_check_y_params,
        cd_cv_fit_params_guard_args,
        cd_cv_fortran_check_x_params,
        cd_cv_initial_copy_x,
        cd_cv_non_reference_copy_x,
        cd_cv_reference_check_x_params,
        cd_cv_reference_validation_copy_x,
    )

    estimator = object()
    params = {"groups": [0, 1]}
    guard_args = cd_cv_fit_params_guard_args(params, estimator)
    assert guard_args[0] == params
    assert guard_args[1] is estimator
    assert guard_args[2] == "fit"

    assert cd_cv_initial_copy_x(True, True) is True
    assert cd_cv_initial_copy_x(True, False) is False

    assert cd_cv_check_y_params(True) == {
        "copy": False,
        "dtype": [np.float64, np.float32],
        "ensure_2d": False,
    }
    assert cd_cv_reference_check_x_params(True) == {
        "accept_sparse": "csc",
        "dtype": [np.float64, np.float32],
        "force_writeable": True,
        "copy": False,
        "accept_large_sparse": False,
    }
    assert cd_cv_fortran_check_x_params(True, False) == {
        "accept_sparse": "csc",
        "dtype": [np.float64, np.float32],
        "order": "F",
        "force_writeable": True,
        "copy": True,
    }
    assert cd_cv_reference_validation_copy_x(True, True, True, False) is False
    assert cd_cv_reference_validation_copy_x(True, False, False, True) is False
    assert cd_cv_reference_validation_copy_x(True, False, False, False) is True
    assert cd_cv_non_reference_copy_x(False) is False


def test_coordinate_descent_cv_validation_prelude_shell_contracts() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_cv_validation_prelude_shell import (
        cd_cv_check_y_params,
        cd_cv_fit_params_guard_args,
        cd_cv_fortran_check_x_params,
        cd_cv_initial_copy_x,
        cd_cv_non_reference_copy_x,
        cd_cv_reference_check_x_params,
    )

    with pytest.raises(ViolationError):
        cd_cv_fit_params_guard_args([], object())  # type: ignore[arg-type]

    with pytest.raises(ViolationError):
        cd_cv_initial_copy_x("yes", True)  # type: ignore[arg-type]

    with pytest.raises(ViolationError):
        cd_cv_check_y_params(False)

    with pytest.raises(ViolationError):
        cd_cv_reference_check_x_params(False)

    with pytest.raises(ViolationError):
        cd_cv_fortran_check_x_params(True, True)

    with pytest.raises(ViolationError):
        cd_cv_non_reference_copy_x(True)
