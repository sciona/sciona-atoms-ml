from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError


def test_coordinate_descent_multitask_validation_prelude_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_multitask_validation_prelude_shell import (
        cd_multitask_check_x_params,
        cd_multitask_check_y_params,
        cd_multitask_consistent_length_args,
        cd_multitask_shape_counts,
        cd_multitask_validate_data_args,
        cd_multitask_validate_data_kwargs,
        cd_multitask_y_astype_dtype,
    )

    assert callable(cd_multitask_check_x_params)
    assert callable(cd_multitask_check_y_params)
    assert callable(cd_multitask_validate_data_args)
    assert callable(cd_multitask_validate_data_kwargs)
    assert callable(cd_multitask_consistent_length_args)
    assert callable(cd_multitask_y_astype_dtype)
    assert callable(cd_multitask_shape_counts)


def test_coordinate_descent_multitask_validation_prelude_shell_matches_sklearn_setup() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_multitask_validation_prelude_shell import (
        cd_multitask_check_x_params,
        cd_multitask_check_y_params,
        cd_multitask_consistent_length_args,
        cd_multitask_shape_counts,
        cd_multitask_validate_data_args,
        cd_multitask_validate_data_kwargs,
        cd_multitask_y_astype_dtype,
    )

    check_X_params = cd_multitask_check_x_params(True, True)
    assert check_X_params == {
        "dtype": [np.float64, np.float32],
        "order": "F",
        "force_writeable": True,
        "copy": True,
    }
    assert cd_multitask_check_x_params(True, False)["copy"] is False

    check_y_params = cd_multitask_check_y_params(object())
    assert check_y_params == {"ensure_2d": False, "order": "F"}

    estimator = object()
    X = object()
    y = np.array([[1, 2], [3, 4]], dtype=np.int64)
    args = cd_multitask_validate_data_args(estimator, X, y)
    assert args == (estimator, X, y)
    assert args[0] is estimator
    assert args[1] is X
    assert args[2] is y

    kwargs = cd_multitask_validate_data_kwargs(check_X_params, check_y_params)
    assert kwargs == {"validate_separately": (check_X_params, check_y_params)}
    assert kwargs["validate_separately"][0] is check_X_params
    assert kwargs["validate_separately"][1] is check_y_params

    length_args = cd_multitask_consistent_length_args(X, y)
    assert length_args[0] is X
    assert length_args[1] is y

    cast_y = cd_multitask_y_astype_dtype(y, np.float32)
    assert cast_y.dtype == np.float32
    assert np.array_equal(cast_y, y.astype(np.float32))

    assert cd_multitask_shape_counts((2, 3), (2, 4)) == (2, 3, 4)


def test_coordinate_descent_multitask_validation_prelude_shell_contracts() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_multitask_validation_prelude_shell import (
        cd_multitask_check_x_params,
        cd_multitask_shape_counts,
        cd_multitask_validate_data_kwargs,
        cd_multitask_y_astype_dtype,
    )

    with pytest.raises(ViolationError):
        cd_multitask_check_x_params(True, "yes")  # type: ignore[arg-type]

    with pytest.raises(ViolationError):
        cd_multitask_validate_data_kwargs([], {})  # type: ignore[arg-type]

    with pytest.raises(ViolationError):
        cd_multitask_y_astype_dtype(np.array([1.0]), object())

    with pytest.raises(ViolationError):
        cd_multitask_shape_counts((2, 3), (4, 1))
