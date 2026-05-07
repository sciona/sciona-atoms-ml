from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError


def test_coordinate_descent_estimator_prefit_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_estimator_prefit_shell import (
        cd_estimator_n_targets,
        cd_estimator_pre_fit_args,
        cd_estimator_pre_fit_kwargs,
        cd_estimator_set_order_args,
        cd_estimator_set_order_required,
        cd_estimator_should_copy,
        cd_estimator_xy_column_vector,
        cd_estimator_xy_column_vector_required,
        cd_estimator_y_column_vector,
        cd_estimator_y_column_vector_required,
    )

    assert callable(cd_estimator_should_copy)
    assert callable(cd_estimator_pre_fit_args)
    assert callable(cd_estimator_pre_fit_kwargs)
    assert callable(cd_estimator_set_order_required)
    assert callable(cd_estimator_set_order_args)
    assert callable(cd_estimator_y_column_vector_required)
    assert callable(cd_estimator_y_column_vector)
    assert callable(cd_estimator_xy_column_vector_required)
    assert callable(cd_estimator_xy_column_vector)
    assert callable(cd_estimator_n_targets)


def test_coordinate_descent_estimator_prefit_shell_matches_sklearn_shells() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_estimator_prefit_shell import (
        cd_estimator_n_targets,
        cd_estimator_pre_fit_args,
        cd_estimator_pre_fit_kwargs,
        cd_estimator_set_order_args,
        cd_estimator_set_order_required,
        cd_estimator_should_copy,
        cd_estimator_xy_column_vector,
        cd_estimator_xy_column_vector_required,
        cd_estimator_y_column_vector,
        cd_estimator_y_column_vector_required,
    )

    assert cd_estimator_should_copy(True, False) is True
    assert cd_estimator_should_copy(True, True) is False

    X = object()
    y = object()
    precompute = object()
    pre_fit_args = cd_estimator_pre_fit_args(X, y, precompute)
    assert pre_fit_args == (X, y, None, precompute)
    assert pre_fit_args[0] is X
    assert pre_fit_args[1] is y
    assert pre_fit_args[3] is precompute

    sample_weight = object()
    assert cd_estimator_pre_fit_kwargs(True, False, True, sample_weight) == {
        "fit_intercept": True,
        "copy": False,
        "check_input": True,
        "sample_weight": sample_weight,
    }

    assert cd_estimator_set_order_required(True, None) is True
    assert cd_estimator_set_order_required(False, sample_weight) is True
    assert cd_estimator_set_order_required(False, None) is False
    set_order_args = cd_estimator_set_order_args(X, y)
    assert set_order_args == (X, y)
    assert set_order_args[0] is X
    assert set_order_args[1] is y

    y_vector = np.array([1.0, 2.0])
    assert cd_estimator_y_column_vector_required(1) is True
    assert np.array_equal(cd_estimator_y_column_vector(y_vector), y_vector[:, np.newaxis])

    Xy_vector = np.array([3.0, 4.0])
    assert cd_estimator_xy_column_vector_required(None) is False
    assert cd_estimator_xy_column_vector_required(Xy_vector) is True
    assert np.array_equal(cd_estimator_xy_column_vector(Xy_vector), Xy_vector[:, np.newaxis])
    assert cd_estimator_n_targets((2, 1)) == 1


def test_coordinate_descent_estimator_prefit_shell_contracts() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_estimator_prefit_shell import (
        cd_estimator_n_targets,
        cd_estimator_pre_fit_kwargs,
        cd_estimator_should_copy,
        cd_estimator_xy_column_vector,
        cd_estimator_y_column_vector,
        cd_estimator_y_column_vector_required,
    )

    with pytest.raises(ViolationError):
        cd_estimator_should_copy(True, "no")  # type: ignore[arg-type]

    with pytest.raises(ViolationError):
        cd_estimator_pre_fit_kwargs(True, False, "yes", None)  # type: ignore[arg-type]

    with pytest.raises(ViolationError):
        cd_estimator_y_column_vector_required(0)

    with pytest.raises(ViolationError):
        cd_estimator_y_column_vector(np.array([[1.0]]))

    with pytest.raises(ViolationError):
        cd_estimator_xy_column_vector(np.array([[1.0]]))

    with pytest.raises(ViolationError):
        cd_estimator_n_targets((2, 0))
