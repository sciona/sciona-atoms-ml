from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError


def test_huber_fit_optimizer_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.linear_model.huber_fit_optimizer_shell import (
        huber_fit_bounds,
        huber_fit_initial_parameters,
        huber_fit_optimizer_payload,
        huber_fit_outlier_handoff_payload,
        huber_fit_result_attributes,
        huber_fit_status2_failure_message,
    )

    assert callable(huber_fit_initial_parameters)
    assert callable(huber_fit_bounds)
    assert callable(huber_fit_optimizer_payload)
    assert callable(huber_fit_status2_failure_message)
    assert callable(huber_fit_result_attributes)
    assert callable(huber_fit_outlier_handoff_payload)


def test_huber_fit_initial_parameters_match_sklearn_branches() -> None:
    from sciona.atoms.ml.sklearn.linear_model.huber_fit_optimizer_shell import huber_fit_initial_parameters

    with_intercept = huber_fit_initial_parameters(3, fit_intercept=True, warm_start=False)
    without_intercept = huber_fit_initial_parameters(3, fit_intercept=False, warm_start=False)

    assert np.array_equal(with_intercept, np.array([0.0, 0.0, 0.0, 0.0, 1.0]))
    assert np.array_equal(without_intercept, np.array([0.0, 0.0, 0.0, 1.0]))

    coef = np.array([0.2, -0.3], dtype=np.float64)
    warm = huber_fit_initial_parameters(
        2,
        fit_intercept=False,
        warm_start=True,
        coef=coef,
        intercept=1.5,
        scale=0.7,
    )
    assert np.array_equal(warm, np.array([0.2, -0.3, 1.5, 0.7]))


def test_huber_fit_bounds_match_sklearn_scale_constraint() -> None:
    from sciona.atoms.ml.sklearn.linear_model.huber_fit_optimizer_shell import huber_fit_bounds

    bounds = huber_fit_bounds(4)

    assert bounds.shape == (4, 2)
    assert np.all(np.isneginf(bounds[:-1, 0]))
    assert np.all(np.isposinf(bounds[:, 1]))
    assert bounds[-1, 0] == np.finfo(np.float64).eps * 10


def test_huber_fit_optimizer_payload_matches_minimize_call_shape() -> None:
    from sciona.atoms.ml.sklearn.linear_model.huber import huber_loss_gradient
    from sciona.atoms.ml.sklearn.linear_model.huber_fit_optimizer_shell import (
        huber_fit_bounds,
        huber_fit_initial_parameters,
        huber_fit_optimizer_payload,
    )

    X = np.array([[1.0, 2.0], [0.5, -1.0]], dtype=np.float64)
    y = np.array([1.5, -0.25], dtype=np.float64)
    sample_weight = np.array([1.0, 2.0], dtype=np.float64)
    parameters = huber_fit_initial_parameters(2, fit_intercept=True, warm_start=False)
    bounds = huber_fit_bounds(parameters.shape[0])

    payload = huber_fit_optimizer_payload(
        huber_loss_gradient,
        parameters,
        X,
        y,
        epsilon=1.35,
        alpha=0.01,
        sample_weight=sample_weight,
        max_iter=100,
        tol=1e-4,
        bounds=bounds,
    )

    assert payload["fun"] is huber_loss_gradient
    assert payload["x0"] is parameters
    assert payload["method"] == "L-BFGS-B"
    assert payload["jac"] is True
    args = payload["args"]
    assert args[0] is X
    assert args[1] is y
    assert args[2] == 1.35
    assert args[3] == 0.01
    assert args[4] is sample_weight
    assert payload["options"] == {"maxiter": 100, "gtol": 1e-4, "iprint": -1}
    assert payload["bounds"] is bounds


def test_huber_fit_status2_failure_message_matches_sklearn_wording() -> None:
    from sciona.atoms.ml.sklearn.linear_model.huber_fit_optimizer_shell import huber_fit_status2_failure_message

    assert huber_fit_status2_failure_message(0, "ok") is None
    assert (
        huber_fit_status2_failure_message(2, "ABNORMAL_TERMINATION_IN_LNSRCH")
        == "HuberRegressor convergence failed: l-BFGS-b solver terminated with ABNORMAL_TERMINATION_IN_LNSRCH"
    )


def test_huber_fit_result_attributes_match_optimizer_tail_unpacking() -> None:
    from sciona.atoms.ml.sklearn.linear_model.huber_fit_optimizer_shell import huber_fit_result_attributes

    with_intercept = huber_fit_result_attributes(
        np.array([0.4, -0.8, 0.15, 1.1], dtype=np.float64),
        2,
        fit_intercept=True,
    )
    without_intercept = huber_fit_result_attributes(
        np.array([0.4, -0.8, 1.1], dtype=np.float64),
        2,
        fit_intercept=False,
    )

    assert np.array_equal(with_intercept["coef"], np.array([0.4, -0.8]))
    assert with_intercept["intercept"] == 0.15
    assert with_intercept["scale"] == 1.1
    assert np.array_equal(without_intercept["coef"], np.array([0.4, -0.8]))
    assert without_intercept["intercept"] == 0.0
    assert without_intercept["scale"] == 1.1


def test_huber_fit_outlier_handoff_payload_reuses_existing_huber_atoms() -> None:
    from sciona.atoms.ml.sklearn.linear_model.huber import huber_linear_residuals, huber_outlier_mask
    from sciona.atoms.ml.sklearn.linear_model.huber_fit_optimizer_shell import huber_fit_outlier_handoff_payload

    X = np.array([[1.0, 2.0], [0.5, -1.0], [3.0, 0.25]], dtype=np.float64)
    y = np.array([1.5, -0.25, 3.0], dtype=np.float64)
    coef = np.array([0.75, -0.5], dtype=np.float64)

    payload = huber_fit_outlier_handoff_payload(X, y, coef, intercept=0.2, scale=0.9, epsilon=1.35)
    residuals = huber_linear_residuals(payload["X"], payload["y"], payload["coef"], intercept=payload["intercept"])
    outliers = huber_outlier_mask(residuals, epsilon=payload["epsilon"], sigma=payload["scale"])

    expected = np.abs(y - X.dot(coef) - 0.2) > 0.9 * 1.35
    assert np.array_equal(outliers, expected)


def test_huber_fit_optimizer_shell_contracts() -> None:
    from sciona.atoms.ml.sklearn.linear_model.huber_fit_optimizer_shell import (
        huber_fit_bounds,
        huber_fit_initial_parameters,
        huber_fit_optimizer_payload,
        huber_fit_outlier_handoff_payload,
        huber_fit_result_attributes,
        huber_fit_status2_failure_message,
    )

    with pytest.raises(ViolationError):
        huber_fit_initial_parameters(0, fit_intercept=True, warm_start=False)

    with pytest.raises(ViolationError):
        huber_fit_initial_parameters(2, fit_intercept=True, warm_start=True, coef=np.ones(2), intercept=0.0, scale=0.0)

    with pytest.raises(ViolationError):
        huber_fit_bounds(0)

    with pytest.raises(ViolationError):
        huber_fit_optimizer_payload(
            object(),
            np.array([0.0, 0.0, 1.0], dtype=np.float64),
            object(),
            object(),
            epsilon=0.5,
            alpha=0.0,
            sample_weight=None,
            max_iter=100,
            tol=1e-4,
            bounds=object(),
        )

    with pytest.raises(ViolationError):
        huber_fit_status2_failure_message(True, "bad")

    with pytest.raises(ViolationError):
        huber_fit_result_attributes(np.array([0.4, -0.8, 1.1], dtype=np.float64), 2, fit_intercept=True)

    with pytest.raises(ViolationError):
        huber_fit_outlier_handoff_payload(object(), object(), object(), intercept=0.0, scale=-1.0, epsilon=1.35)
