from __future__ import annotations

from types import MethodType

import numpy as np
import pytest
from icontract import ViolationError
from sklearn.linear_model import SGDOneClassSVM


def test_sgd_one_class_fit_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.linear_model.sgd_one_class_fit_shell import (
        sgd_one_class_average_active,
        sgd_one_class_average_buffers,
        sgd_one_class_fit_one_class_payload,
        sgd_one_class_fixed_solver_context,
        sgd_one_class_intercept_from_offset,
        sgd_one_class_offset_from_intercept,
        sgd_one_class_parameter_allocation_payload,
        sgd_one_class_partial_fit_result,
        sgd_one_class_target,
        sgd_one_class_time_step_after_fit,
        sgd_one_class_validation_sample_mask,
    )

    assert callable(sgd_one_class_target)
    assert callable(sgd_one_class_fixed_solver_context)
    assert callable(sgd_one_class_validation_sample_mask)
    assert callable(sgd_one_class_intercept_from_offset)
    assert callable(sgd_one_class_offset_from_intercept)
    assert callable(sgd_one_class_time_step_after_fit)
    assert callable(sgd_one_class_average_active)
    assert callable(sgd_one_class_average_buffers)
    assert callable(sgd_one_class_parameter_allocation_payload)
    assert callable(sgd_one_class_fit_one_class_payload)
    assert callable(sgd_one_class_partial_fit_result)


def test_sgd_one_class_target_mask_offset_and_average_helpers() -> None:
    from sciona.atoms.ml.sklearn.linear_model.sgd_one_class_fit_shell import (
        sgd_one_class_average_active,
        sgd_one_class_fixed_solver_context,
        sgd_one_class_intercept_from_offset,
        sgd_one_class_offset_from_intercept,
        sgd_one_class_target,
        sgd_one_class_time_step_after_fit,
        sgd_one_class_validation_sample_mask,
    )

    X = np.array([[1.0, 0.5], [2.0, 1.5], [3.0, 2.5]], dtype=np.float32)
    sample_weight = np.array([1.0, 0.0, 2.5], dtype=np.float32)
    offset = np.array([0.25], dtype=np.float64)

    target = sgd_one_class_target(X)
    intercept = sgd_one_class_intercept_from_offset(offset)
    t_after = sgd_one_class_time_step_after_fit(1.0, n_iter=3, n_samples=X.shape[0])

    assert target.dtype == X.dtype
    assert target.flags.c_contiguous
    np.testing.assert_array_equal(target, np.ones(X.shape[0], dtype=X.dtype))
    np.testing.assert_array_equal(sgd_one_class_validation_sample_mask(sample_weight), sample_weight > 0)
    assert sgd_one_class_fixed_solver_context() == (1, 1, 1)
    np.testing.assert_allclose(intercept, 1 - offset)
    np.testing.assert_allclose(sgd_one_class_offset_from_intercept(intercept), offset)
    assert t_after == 10.0
    assert sgd_one_class_average_active(True, t_after) is True
    assert sgd_one_class_average_active(12, t_after) is False


def test_sgd_one_class_allocation_and_delegation_payloads_match_partial_fit() -> None:
    from sciona.atoms.ml.sklearn.linear_model.sgd_one_class_fit_shell import (
        sgd_one_class_average_buffers,
        sgd_one_class_fit_one_class_payload,
        sgd_one_class_parameter_allocation_payload,
        sgd_one_class_partial_fit_result,
    )

    estimator = SGDOneClassSVM(average=True, learning_rate="constant", eta0=0.01)
    X = np.array([[0.0, 1.0], [1.0, 0.0], [2.0, 2.0]], dtype=np.float64)
    coef_init = np.array([0.1, -0.1], dtype=np.float64)
    offset_init = np.array([0.25], dtype=np.float64)
    sample_weight = np.array([1.0, 0.0, 2.0], dtype=np.float64)
    captured: dict[str, object] = {}

    def fake_allocate_parameter_mem(self: SGDOneClassSVM, **kwargs: object) -> None:
        captured["allocation"] = kwargs
        self.coef_ = np.zeros(int(kwargs["n_features"]), dtype=kwargs["input_dtype"])
        self.offset_ = np.zeros(1, dtype=kwargs["input_dtype"])

    def fake_fit_one_class(self: SGDOneClassSVM, X_arg: object, **kwargs: object) -> None:
        captured["X"] = X_arg
        captured["fit_one_class"] = kwargs
        self.n_iter_ = 0
        return None

    estimator._allocate_parameter_mem = MethodType(fake_allocate_parameter_mem, estimator)
    estimator._fit_one_class = MethodType(fake_fit_one_class, estimator)

    result = estimator._partial_fit(
        X,
        alpha=0.1,
        C=1.0,
        loss="hinge",
        learning_rate=estimator.learning_rate,
        max_iter=5,
        sample_weight=sample_weight,
        coef_init=coef_init,
        offset_init=offset_init,
    )

    allocation_payload = sgd_one_class_parameter_allocation_payload(
        X.shape[1],
        X.dtype,
        coef_init=coef_init,
        offset_init=offset_init,
    )
    fit_payload = sgd_one_class_fit_one_class_payload(
        captured["X"],
        alpha=0.1,
        C=1.0,
        learning_rate=estimator.learning_rate,
        sample_weight=captured["fit_one_class"]["sample_weight"],
        max_iter=5,
    )
    average_buffers = sgd_one_class_average_buffers(X.shape[1], X.dtype)

    assert sgd_one_class_partial_fit_result(result) is estimator
    assert captured["allocation"] == allocation_payload
    assert captured["fit_one_class"] == {key: value for key, value in fit_payload.items() if key != "X"}
    np.testing.assert_array_equal(estimator._average_coef, average_buffers["average_coef"])
    np.testing.assert_array_equal(estimator._average_intercept, average_buffers["average_intercept"])


def test_sgd_one_class_payloads_preserve_optional_objects() -> None:
    from sciona.atoms.ml.sklearn.linear_model.sgd_one_class_fit_shell import (
        sgd_one_class_fit_one_class_payload,
        sgd_one_class_parameter_allocation_payload,
    )

    X = object()
    coef_init = object()
    offset_init = object()
    sample_weight = object()

    allocation_payload = sgd_one_class_parameter_allocation_payload(
        3,
        np.float64,
        coef_init=coef_init,
        offset_init=offset_init,
    )
    fit_payload = sgd_one_class_fit_one_class_payload(
        X,
        alpha=0.2,
        C=1.0,
        learning_rate="optimal",
        sample_weight=sample_weight,
        max_iter=7,
    )

    assert allocation_payload["n_classes"] == 1
    assert allocation_payload["n_features"] == 3
    assert allocation_payload["input_dtype"] == np.dtype(np.float64)
    assert allocation_payload["coef_init"] is coef_init
    assert allocation_payload["intercept_init"] is offset_init
    assert allocation_payload["one_class"] == 1
    assert fit_payload["X"] is X
    assert fit_payload["sample_weight"] is sample_weight


def test_sgd_one_class_fit_shell_contracts() -> None:
    from sciona.atoms.ml.sklearn.linear_model.sgd_one_class_fit_shell import (
        sgd_one_class_average_active,
        sgd_one_class_average_buffers,
        sgd_one_class_fit_one_class_payload,
        sgd_one_class_intercept_from_offset,
        sgd_one_class_parameter_allocation_payload,
        sgd_one_class_partial_fit_result,
        sgd_one_class_target,
        sgd_one_class_time_step_after_fit,
        sgd_one_class_validation_sample_mask,
    )

    with pytest.raises(ViolationError):
        sgd_one_class_target(np.array([1.0, 2.0], dtype=np.float64))

    with pytest.raises(ViolationError):
        sgd_one_class_validation_sample_mask(np.array([1.0, np.nan], dtype=np.float64))

    with pytest.raises(ViolationError):
        sgd_one_class_intercept_from_offset(np.array([np.inf]))

    with pytest.raises(ViolationError):
        sgd_one_class_time_step_after_fit(1.0, n_iter=-1, n_samples=3)

    with pytest.raises(ViolationError):
        sgd_one_class_average_active("yes", 4.0)

    with pytest.raises(ViolationError):
        sgd_one_class_average_buffers(0, np.float64)

    with pytest.raises(ViolationError):
        sgd_one_class_parameter_allocation_payload(2, np.int64)

    with pytest.raises(ViolationError):
        sgd_one_class_fit_one_class_payload(object(), alpha=-0.1, C=1.0, learning_rate="optimal", sample_weight=None, max_iter=5)

    with pytest.raises(ViolationError):
        sgd_one_class_partial_fit_result(None)
