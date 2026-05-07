from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError


def test_coordinate_descent_estimator_validation_prelude_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_estimator_validation_prelude_shell import (
        cd_estimator_alpha_zero_warning_message,
        cd_estimator_alpha_zero_warning_required,
        cd_estimator_check_array_y_kwargs,
        cd_estimator_shape_counts,
        cd_estimator_validate_data_args,
        cd_estimator_validate_data_kwargs,
        cd_estimator_x_copied,
    )

    assert callable(cd_estimator_alpha_zero_warning_required)
    assert callable(cd_estimator_alpha_zero_warning_message)
    assert callable(cd_estimator_x_copied)
    assert callable(cd_estimator_validate_data_args)
    assert callable(cd_estimator_validate_data_kwargs)
    assert callable(cd_estimator_check_array_y_kwargs)
    assert callable(cd_estimator_shape_counts)


def test_coordinate_descent_estimator_validation_prelude_shell_matches_sklearn_shells() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_estimator_validation_prelude_shell import (
        cd_estimator_alpha_zero_warning_message,
        cd_estimator_alpha_zero_warning_required,
        cd_estimator_check_array_y_kwargs,
        cd_estimator_shape_counts,
        cd_estimator_validate_data_args,
        cd_estimator_validate_data_kwargs,
        cd_estimator_x_copied,
    )

    assert cd_estimator_alpha_zero_warning_required(0.0) is True
    assert cd_estimator_alpha_zero_warning_required(0.5) is False
    assert cd_estimator_alpha_zero_warning_message(0.0) == (
        "With alpha=0, this algorithm does not converge "
        "well. You are advised to use the LinearRegression "
        "estimator"
    )

    assert cd_estimator_x_copied(True, True, True) is True
    assert cd_estimator_x_copied(True, True, False) is False
    assert cd_estimator_x_copied(True, False, True) is False

    estimator = object()
    X = object()
    y = object()
    args = cd_estimator_validate_data_args(estimator, X, y)
    assert args == (estimator, X, y)
    assert args[0] is estimator
    assert args[1] is X
    assert args[2] is y

    validate_kwargs = cd_estimator_validate_data_kwargs(True)
    assert validate_kwargs == {
        "accept_sparse": "csc",
        "order": "F",
        "dtype": [np.float64, np.float32],
        "force_writeable": True,
        "accept_large_sparse": False,
        "copy": True,
        "multi_output": True,
        "y_numeric": True,
    }
    assert cd_estimator_check_array_y_kwargs(np.float64) == {
        "order": "F",
        "copy": False,
        "dtype": np.float64,
        "ensure_2d": False,
    }
    assert cd_estimator_shape_counts((4, 3)) == (4, 3)


def test_coordinate_descent_estimator_validation_prelude_shell_contracts() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_estimator_validation_prelude_shell import (
        cd_estimator_alpha_zero_warning_required,
        cd_estimator_check_array_y_kwargs,
        cd_estimator_shape_counts,
        cd_estimator_validate_data_kwargs,
        cd_estimator_x_copied,
    )

    with pytest.raises(ViolationError):
        cd_estimator_alpha_zero_warning_required("0")  # type: ignore[arg-type]

    with pytest.raises(ViolationError):
        cd_estimator_x_copied(True, True, "yes")  # type: ignore[arg-type]

    with pytest.raises(ViolationError):
        cd_estimator_validate_data_kwargs(None)  # type: ignore[arg-type]

    with pytest.raises(ViolationError):
        cd_estimator_check_array_y_kwargs(None)

    with pytest.raises(ViolationError):
        cd_estimator_shape_counts((4, 0))
