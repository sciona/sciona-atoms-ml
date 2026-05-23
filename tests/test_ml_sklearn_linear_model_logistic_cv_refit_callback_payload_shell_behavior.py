from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError


def test_logistic_cv_refit_callback_payload_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.linear_model.logistic_cv_refit_callback_payload_shell import (
        logistic_cv_refit_first_weight,
        logistic_cv_refit_path_call,
        logistic_cv_refit_path_kwargs,
        logistic_cv_refit_single_Cs,
        logistic_cv_refit_verbose,
    )

    assert callable(logistic_cv_refit_single_Cs)
    assert callable(logistic_cv_refit_verbose)
    assert callable(logistic_cv_refit_path_kwargs)
    assert callable(logistic_cv_refit_path_call)
    assert callable(logistic_cv_refit_first_weight)


def test_logistic_cv_refit_single_Cs_and_verbose_match_source_transforms() -> None:
    from sciona.atoms.ml.sklearn.linear_model.logistic_cv_refit_callback_payload_shell import (
        logistic_cv_refit_single_Cs,
        logistic_cv_refit_verbose,
    )

    C_value = np.float64(0.5)
    Cs = logistic_cv_refit_single_Cs(C_value)

    assert Cs == [C_value]
    assert Cs[0] is C_value
    assert logistic_cv_refit_verbose(0) == 0
    assert logistic_cv_refit_verbose(1) == 0
    assert logistic_cv_refit_verbose(4) == 3


def test_logistic_cv_refit_path_kwargs_match_sklearn_refit_call_payload() -> None:
    from sciona.atoms.ml.sklearn.linear_model.logistic_cv_refit_callback_payload_shell import logistic_cv_refit_path_kwargs

    pos_class = object()
    C_value = np.float64(1.25)
    solver = "lbfgs"
    coef_init = object()
    penalty = "l2"
    class_weight = {"a": 1.0}
    random_state = object()
    max_squared_sum = object()
    sample_weight = object()
    l1_ratio = None

    kwargs = logistic_cv_refit_path_kwargs(
        pos_class,
        C_value,
        solver,
        True,
        coef_init,
        100,
        1e-4,
        penalty,
        class_weight,
        "ovr",
        3,
        random_state,
        max_squared_sum,
        sample_weight,
        l1_ratio,
    )

    assert list(kwargs) == [
        "pos_class",
        "Cs",
        "solver",
        "fit_intercept",
        "coef",
        "max_iter",
        "tol",
        "penalty",
        "class_weight",
        "multi_class",
        "verbose",
        "random_state",
        "check_input",
        "max_squared_sum",
        "sample_weight",
        "l1_ratio",
    ]
    assert kwargs["pos_class"] is pos_class
    assert kwargs["Cs"] == [C_value]
    assert kwargs["Cs"][0] is C_value
    assert kwargs["solver"] is solver
    assert kwargs["fit_intercept"] is True
    assert kwargs["coef"] is coef_init
    assert kwargs["max_iter"] == 100
    assert kwargs["tol"] == 1e-4
    assert kwargs["penalty"] is penalty
    assert kwargs["class_weight"] is class_weight
    assert kwargs["multi_class"] == "ovr"
    assert kwargs["verbose"] == 2
    assert kwargs["random_state"] is random_state
    assert kwargs["check_input"] is False
    assert kwargs["max_squared_sum"] is max_squared_sum
    assert kwargs["sample_weight"] is sample_weight
    assert kwargs["l1_ratio"] is None


def test_logistic_cv_refit_path_kwargs_match_multinomial_payload() -> None:
    from sciona.atoms.ml.sklearn.linear_model.logistic_cv_refit_callback_payload_shell import logistic_cv_refit_path_kwargs

    kwargs = logistic_cv_refit_path_kwargs(
        None,
        0.75,
        "saga",
        False,
        np.ones((3, 4), dtype=np.float64),
        25,
        0.0,
        "elasticnet",
        None,
        "multinomial",
        0,
        123,
        None,
        np.ones(5, dtype=np.float64),
        0.5,
    )

    assert kwargs["pos_class"] is None
    assert kwargs["Cs"] == [0.75]
    assert kwargs["verbose"] == 0
    assert kwargs["check_input"] is False
    assert kwargs["l1_ratio"] == 0.5


def test_logistic_cv_refit_path_call_preserves_solver_boundary_payload_identity() -> None:
    from sciona.atoms.ml.sklearn.linear_model.logistic_cv_refit_callback_payload_shell import (
        logistic_cv_refit_path_call,
        logistic_cv_refit_path_kwargs,
    )

    X = object()
    y = object()
    kwargs = logistic_cv_refit_path_kwargs(None, 1.0, "lbfgs", True, object(), 10, 1e-3, "l2", None, "multinomial", 1, None, None, None, None)

    call_payload = logistic_cv_refit_path_call(X, y, kwargs)

    assert call_payload == (X, y, kwargs)
    assert call_payload[0] is X
    assert call_payload[1] is y
    assert call_payload[2] is kwargs


def test_logistic_cv_refit_first_weight_matches_source_result_extraction() -> None:
    from sciona.atoms.ml.sklearn.linear_model.logistic_cv_refit_callback_payload_shell import logistic_cv_refit_first_weight

    weights = np.arange(6, dtype=np.float64).reshape(2, 3)
    first = logistic_cv_refit_first_weight(weights)

    np.testing.assert_array_equal(first, weights[0])
    assert first is weights[0] or np.array_equal(first, weights[0])
    assert logistic_cv_refit_first_weight(["a", "b"]) == "a"


def test_logistic_cv_refit_callback_payload_shell_contracts() -> None:
    from sciona.atoms.ml.sklearn.linear_model.logistic_cv_refit_callback_payload_shell import (
        logistic_cv_refit_first_weight,
        logistic_cv_refit_path_call,
        logistic_cv_refit_path_kwargs,
        logistic_cv_refit_single_Cs,
        logistic_cv_refit_verbose,
    )

    with pytest.raises(ViolationError):
        logistic_cv_refit_single_Cs(0.0)

    with pytest.raises(ViolationError):
        logistic_cv_refit_verbose(True)  # type: ignore[arg-type]

    with pytest.raises(ViolationError):
        logistic_cv_refit_path_kwargs(None, 1.0, "lbfgs", "yes", object(), 10, 1e-3, "l2", None, "ovr", 0, None, None, None, None)  # type: ignore[arg-type]

    with pytest.raises(ViolationError):
        logistic_cv_refit_path_kwargs(None, 1.0, "lbfgs", True, object(), 0, 1e-3, "l2", None, "ovr", 0, None, None, None, None)

    with pytest.raises(ViolationError):
        logistic_cv_refit_path_kwargs(None, 1.0, "lbfgs", True, object(), 1, -1e-3, "l2", None, "ovr", 0, None, None, None, None)

    with pytest.raises(ViolationError):
        logistic_cv_refit_path_kwargs(None, 1.0, "lbfgs", True, object(), 1, 1e-3, "l2", None, "bad", 0, None, None, None, None)

    with pytest.raises(ViolationError):
        logistic_cv_refit_path_call(object(), object(), {"bad": object()})

    with pytest.raises(ViolationError):
        logistic_cv_refit_first_weight([])
