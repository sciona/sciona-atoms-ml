from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError


def test_coordinate_descent_estimator_sample_weight_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_estimator_sample_weight_shell import (
        cd_estimator_check_sample_weight_args,
        cd_estimator_check_sample_weight_kwargs,
        cd_estimator_checked_sample_weight,
        cd_estimator_sample_weight_after_scalar_guard,
        cd_estimator_sample_weight_check_required,
        cd_estimator_sample_weight_rescale_factor,
        cd_estimator_sample_weight_rescaled,
    )

    assert callable(cd_estimator_sample_weight_after_scalar_guard)
    assert callable(cd_estimator_sample_weight_check_required)
    assert callable(cd_estimator_check_sample_weight_args)
    assert callable(cd_estimator_check_sample_weight_kwargs)
    assert callable(cd_estimator_checked_sample_weight)
    assert callable(cd_estimator_sample_weight_rescale_factor)
    assert callable(cd_estimator_sample_weight_rescaled)


def test_coordinate_descent_estimator_sample_weight_shell_matches_sklearn_shells() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_estimator_sample_weight_shell import (
        cd_estimator_check_sample_weight_args,
        cd_estimator_check_sample_weight_kwargs,
        cd_estimator_checked_sample_weight,
        cd_estimator_sample_weight_after_scalar_guard,
        cd_estimator_sample_weight_check_required,
        cd_estimator_sample_weight_rescale_factor,
        cd_estimator_sample_weight_rescaled,
    )

    assert cd_estimator_sample_weight_after_scalar_guard(2.0) is None
    sample_weight = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    assert cd_estimator_sample_weight_after_scalar_guard(sample_weight) is sample_weight

    assert cd_estimator_sample_weight_check_required(sample_weight, True) is True
    assert cd_estimator_sample_weight_check_required(sample_weight, False) is False
    assert cd_estimator_sample_weight_check_required(None, True) is False

    X = object()
    args = cd_estimator_check_sample_weight_args(sample_weight, X)
    assert args == (sample_weight, X)
    assert args[0] is sample_weight
    assert args[1] is X
    assert cd_estimator_check_sample_weight_kwargs("float32") == {"dtype": "float32"}
    checked = object()
    assert cd_estimator_checked_sample_weight(checked) is checked

    factor = cd_estimator_sample_weight_rescale_factor(sample_weight, 3)
    assert factor == pytest.approx(0.5)
    rescaled = cd_estimator_sample_weight_rescaled(sample_weight, 3)
    assert np.allclose(rescaled, np.array([0.5, 1.0, 1.5]))
    assert np.sum(rescaled) == pytest.approx(3.0)


def test_coordinate_descent_estimator_sample_weight_shell_contracts() -> None:
    from sciona.atoms.ml.sklearn.linear_model.coordinate_descent_estimator_sample_weight_shell import (
        cd_estimator_check_sample_weight_args,
        cd_estimator_sample_weight_check_required,
        cd_estimator_sample_weight_rescale_factor,
        cd_estimator_sample_weight_rescaled,
    )

    with pytest.raises(ViolationError):
        cd_estimator_sample_weight_check_required(object(), "yes")  # type: ignore[arg-type]

    with pytest.raises(ViolationError):
        cd_estimator_check_sample_weight_args(None, object())

    with pytest.raises(ViolationError):
        cd_estimator_sample_weight_rescale_factor(np.array([0.0, 0.0]), 2)

    with pytest.raises(ViolationError):
        cd_estimator_sample_weight_rescaled(np.array([1.0, 2.0]), 0)
