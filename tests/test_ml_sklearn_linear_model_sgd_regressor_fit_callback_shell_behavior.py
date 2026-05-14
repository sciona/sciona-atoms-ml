from __future__ import annotations

from types import MethodType

import numpy as np
import pytest
from icontract import ViolationError
from sklearn.linear_model import SGDRegressor


def test_sgd_regressor_fit_callback_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.linear_model.sgd_regressor_fit_callback_shell import (
        sgd_regressor_fit_c_value,
        sgd_regressor_fit_callback_payload,
        sgd_regressor_fit_more_validate_params_result,
        sgd_regressor_fit_result,
    )

    assert callable(sgd_regressor_fit_more_validate_params_result)
    assert callable(sgd_regressor_fit_c_value)
    assert callable(sgd_regressor_fit_callback_payload)
    assert callable(sgd_regressor_fit_result)


def test_sgd_regressor_fit_callback_payload_matches_sklearn_delegation() -> None:
    from sciona.atoms.ml.sklearn.linear_model.sgd_regressor_fit_callback_shell import (
        sgd_regressor_fit_callback_payload,
    )

    reg = SGDRegressor(alpha=0.002, loss="huber", learning_rate="constant", eta0=0.01)
    X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
    y = np.array([0.5, 1.5], dtype=np.float64)
    coef_init = np.array([0.1, -0.1], dtype=np.float64)
    intercept_init = np.array([0.0], dtype=np.float64)
    sample_weight = np.array([1.0, 2.0], dtype=np.float64)
    captured: dict[str, object] = {}

    def fake_more_validate_params(self: SGDRegressor) -> None:
        captured["validated"] = self
        return None

    def fake_fit(self: SGDRegressor, X_arg: object, y_arg: object, **kwargs: object) -> SGDRegressor:
        captured["X"] = X_arg
        captured["y"] = y_arg
        captured.update(kwargs)
        return self

    reg._more_validate_params = MethodType(fake_more_validate_params, reg)
    reg._fit = MethodType(fake_fit, reg)

    result = reg.fit(X, y, coef_init=coef_init, intercept_init=intercept_init, sample_weight=sample_weight)
    payload = sgd_regressor_fit_callback_payload(
        X,
        y,
        alpha=reg.alpha,
        loss=reg.loss,
        learning_rate=reg.learning_rate,
        coef_init=coef_init,
        intercept_init=intercept_init,
        sample_weight=sample_weight,
    )

    assert result is reg
    assert captured["validated"] is reg
    for key, value in payload.items():
        if key in {"X", "y", "coef_init", "intercept_init", "sample_weight"}:
            assert captured[key] is value
        else:
            assert captured[key] == value


def test_sgd_regressor_fit_callback_shell_identities_and_fixed_c() -> None:
    from sciona.atoms.ml.sklearn.linear_model.sgd_regressor_fit_callback_shell import (
        sgd_regressor_fit_c_value,
        sgd_regressor_fit_more_validate_params_result,
        sgd_regressor_fit_result,
    )

    fit_result = object()

    assert sgd_regressor_fit_more_validate_params_result(None) is None
    assert sgd_regressor_fit_c_value(0.001) == 1.0
    assert sgd_regressor_fit_result(fit_result) is fit_result


def test_sgd_regressor_fit_callback_payload_preserves_optional_objects() -> None:
    from sciona.atoms.ml.sklearn.linear_model.sgd_regressor_fit_callback_shell import (
        sgd_regressor_fit_callback_payload,
    )

    X = object()
    y = object()
    coef_init = object()
    intercept_init = object()
    sample_weight = object()

    payload = sgd_regressor_fit_callback_payload(
        X,
        y,
        alpha=0.5,
        loss="squared_error",
        learning_rate="optimal",
        coef_init=coef_init,
        intercept_init=intercept_init,
        sample_weight=sample_weight,
    )

    assert payload["X"] is X
    assert payload["y"] is y
    assert payload["alpha"] == 0.5
    assert payload["C"] == 1.0
    assert payload["loss"] == "squared_error"
    assert payload["learning_rate"] == "optimal"
    assert payload["coef_init"] is coef_init
    assert payload["intercept_init"] is intercept_init
    assert payload["sample_weight"] is sample_weight


def test_sgd_regressor_fit_callback_shell_contracts() -> None:
    from sciona.atoms.ml.sklearn.linear_model.sgd_regressor_fit_callback_shell import (
        sgd_regressor_fit_c_value,
        sgd_regressor_fit_callback_payload,
        sgd_regressor_fit_result,
    )

    with pytest.raises(ViolationError):
        sgd_regressor_fit_c_value(-0.1)

    with pytest.raises(ViolationError):
        sgd_regressor_fit_callback_payload(None, object(), alpha=0.1, loss="squared_error", learning_rate="optimal")

    with pytest.raises(ViolationError):
        sgd_regressor_fit_callback_payload(object(), object(), alpha=float("nan"), loss="squared_error", learning_rate="optimal")

    with pytest.raises(ViolationError):
        sgd_regressor_fit_callback_payload(object(), object(), alpha=0.1, loss="", learning_rate="optimal")

    with pytest.raises(ViolationError):
        sgd_regressor_fit_result(None)
