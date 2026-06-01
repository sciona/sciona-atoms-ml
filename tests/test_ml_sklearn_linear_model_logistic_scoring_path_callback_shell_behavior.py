from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError
from sklearn.linear_model import _logistic as logistic_module
from sklearn.utils.validation import _check_sample_weight


def test_logistic_scoring_path_callback_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.linear_model.logistic_scoring_path_callback_shell import (
        logistic_scoring_classes,
        logistic_scoring_coef_intercept_state,
        logistic_scoring_fold_split,
        logistic_scoring_path_call,
        logistic_scoring_path_kwargs,
        logistic_scoring_positive_y_test,
        logistic_scoring_result_tuple,
        logistic_scoring_sample_weight_split,
        logistic_scoring_score_call_payload,
        logistic_scoring_score_params,
        logistic_scoring_temp_log_reg_kwargs,
    )

    assert callable(logistic_scoring_fold_split)
    assert callable(logistic_scoring_sample_weight_split)
    assert callable(logistic_scoring_path_kwargs)
    assert callable(logistic_scoring_path_call)
    assert callable(logistic_scoring_temp_log_reg_kwargs)
    assert callable(logistic_scoring_classes)
    assert callable(logistic_scoring_positive_y_test)
    assert callable(logistic_scoring_coef_intercept_state)
    assert callable(logistic_scoring_score_params)
    assert callable(logistic_scoring_score_call_payload)
    assert callable(logistic_scoring_result_tuple)


def test_logistic_scoring_path_split_and_solver_payload_match_sklearn(monkeypatch: pytest.MonkeyPatch) -> None:
    from sciona.atoms.ml.sklearn.linear_model.logistic_scoring_path_callback_shell import (
        logistic_scoring_fold_split,
        logistic_scoring_path_kwargs,
        logistic_scoring_sample_weight_split,
    )

    X = np.array([[0.0, 1.0], [1.0, 1.0], [2.0, 0.0], [3.0, 1.0]], dtype=np.float64)
    y = np.array([0, 1, 0, 1])
    train = np.array([0, 2, 3])
    test = np.array([1, 3])
    sample_weight = np.array([1, 2, 3, 4], dtype=np.int64)
    checked_sample_weight = _check_sample_weight(sample_weight, X)
    Cs = np.array([0.5, 1.5], dtype=np.float64)
    coefs = np.array([[0.2, -0.3, 0.1], [0.4, 0.5, -0.2]], dtype=np.float64)
    n_iter = np.array([3, 4], dtype=np.int32)
    captured: dict[str, object] = {}

    def fake_path(X_arg: object, y_arg: object, **kwargs: object) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        captured["X_train"] = X_arg
        captured["y_train"] = y_arg
        captured.update(kwargs)
        return coefs, Cs, n_iter

    score_calls: list[dict[str, object]] = []

    def fake_score(self: object, X_arg: object, y_arg: object, **kwargs: object) -> float:
        score_calls.append({"coef_": self.coef_.copy(), "intercept_": np.array(self.intercept_), "X": X_arg, "y": y_arg, "kwargs": kwargs})
        return 0.5 + 0.25 * len(score_calls)

    monkeypatch.setattr(logistic_module, "_logistic_regression_path", fake_path)
    monkeypatch.setattr(logistic_module, "get_scorer", lambda scoring: None)
    monkeypatch.setattr(logistic_module.LogisticRegression, "score", fake_score)

    result = logistic_module._log_reg_scoring_path(
        X,
        y,
        train,
        test,
        pos_class=1,
        Cs=Cs,
        scoring=None,
        fit_intercept=True,
        max_iter=25,
        tol=1e-4,
        class_weight=None,
        verbose=2,
        solver="lbfgs",
        penalty="l2",
        dual=False,
        intercept_scaling=1.0,
        multi_class="ovr",
        random_state=13,
        max_squared_sum=None,
        sample_weight=sample_weight,
        l1_ratio=None,
        score_params=None,
    )
    X_train, X_test, y_train, y_test = logistic_scoring_fold_split(X, y, train, test)
    sw_train, sw_test = logistic_scoring_sample_weight_split(checked_sample_weight, train, test)
    kwargs = logistic_scoring_path_kwargs(
        Cs,
        None,
        True,
        "lbfgs",
        25,
        None,
        1,
        "ovr",
        1e-4,
        2,
        False,
        "l2",
        1.0,
        13,
        None,
        sw_train,
    )

    assert result[0] is coefs
    assert result[1] is Cs
    assert result[3] is n_iter
    np.testing.assert_array_equal(captured["X_train"], X_train)
    np.testing.assert_array_equal(captured["y_train"], y_train)
    for key, value in kwargs.items():
        if key == "sample_weight":
            np.testing.assert_array_equal(captured[key], value)
        elif key in {"Cs", "l1_ratio", "class_weight", "pos_class", "random_state", "max_squared_sum"}:
            assert captured[key] is value
        else:
            assert captured[key] == value
    assert captured["check_input"] is False
    assert len(score_calls) == 2
    np.testing.assert_array_equal(score_calls[0]["X"], X_test)
    np.testing.assert_array_equal(score_calls[0]["y"], np.array([1.0, 1.0]))
    np.testing.assert_array_equal(score_calls[0]["kwargs"]["sample_weight"], sw_test)  # type: ignore[index]
    np.testing.assert_array_equal(y_test, y[test])


def test_logistic_scoring_path_state_and_recode_helpers() -> None:
    from sciona.atoms.ml.sklearn.linear_model.logistic_scoring_path_callback_shell import (
        logistic_scoring_classes,
        logistic_scoring_coef_intercept_state,
        logistic_scoring_positive_y_test,
        logistic_scoring_temp_log_reg_kwargs,
    )

    y_train = np.array([2, 1, 2, 3])

    assert logistic_scoring_temp_log_reg_kwargs("lbfgs", "ovr") == {"solver": "lbfgs", "multi_class": "ovr"}
    np.testing.assert_array_equal(logistic_scoring_classes("ovr", y_train), np.array([-1, 1]))
    np.testing.assert_array_equal(logistic_scoring_classes("multinomial", y_train), np.array([1, 2, 3]))
    assert logistic_scoring_positive_y_test(y_train, None) is y_train
    np.testing.assert_array_equal(logistic_scoring_positive_y_test(y_train, 2), np.array([1.0, -1.0, 1.0, -1.0]))

    coef, intercept = logistic_scoring_coef_intercept_state(np.array([0.25, 0.5, -1.0]), "ovr", True)
    np.testing.assert_array_equal(coef, np.array([[0.25, 0.5]]))
    np.testing.assert_array_equal(intercept, np.array([-1.0]))

    multi_coef, multi_intercept = logistic_scoring_coef_intercept_state(np.array([[1.0, 2.0], [3.0, 4.0]]), "multinomial", False)
    np.testing.assert_array_equal(multi_coef, np.array([[1.0, 2.0], [3.0, 4.0]]))
    assert multi_intercept == 0.0


def test_logistic_scoring_path_score_payload_and_result_tuple() -> None:
    from sciona.atoms.ml.sklearn.linear_model.logistic_scoring_path_callback_shell import (
        logistic_scoring_result_tuple,
        logistic_scoring_score_call_payload,
        logistic_scoring_score_params,
    )

    X = np.arange(12, dtype=np.float64).reshape(4, 3)
    X_test = X[[1, 3]]
    y_test = np.array([1, -1])
    test = np.array([1, 3])
    estimator_state = (np.array([[0.1, 0.2, 0.3]]), np.array([0.0]))
    sw_test = np.array([2.0, 4.0])

    default_payload = logistic_scoring_score_call_payload(None, estimator_state, X_test, y_test, sw_test, {})
    assert default_payload[0] == "estimator_score"
    assert default_payload[1] is estimator_state
    assert default_payload[2] is X_test
    assert default_payload[3] is y_test
    assert default_payload[4]["sample_weight"] is sw_test  # type: ignore[index]

    score_params = {"sample_weight": np.array([10.0, 20.0, 30.0, 40.0]), "groups": np.array(["a", "b", "c", "d"])}
    checked_params = logistic_scoring_score_params(X, score_params, test)
    scorer = object()
    scorer_payload = logistic_scoring_score_call_payload(scorer, estimator_state, X_test, y_test, None, checked_params)
    assert scorer_payload[0] == "scorer"
    assert scorer_payload[1] is scorer
    np.testing.assert_array_equal(scorer_payload[5]["sample_weight"], np.array([20.0, 40.0]))  # type: ignore[index]
    np.testing.assert_array_equal(scorer_payload[5]["groups"], np.array(["b", "d"]))  # type: ignore[index]

    coefs = object()
    Cs = object()
    n_iter = object()
    result = logistic_scoring_result_tuple(coefs, Cs, [0.25, 0.75], n_iter)
    assert result[0] is coefs
    assert result[1] is Cs
    np.testing.assert_array_equal(result[2], np.array([0.25, 0.75]))
    assert result[3] is n_iter


def test_logistic_scoring_path_call_and_contracts() -> None:
    from sciona.atoms.ml.sklearn.linear_model.logistic_scoring_path_callback_shell import (
        logistic_scoring_classes,
        logistic_scoring_coef_intercept_state,
        logistic_scoring_fold_split,
        logistic_scoring_path_call,
        logistic_scoring_path_kwargs,
        logistic_scoring_score_call_payload,
        logistic_scoring_score_params,
    )

    kwargs = logistic_scoring_path_kwargs(np.array([1.0]), None, True, "lbfgs", 1, None, None, "ovr", 1e-4, 0, False, "l2", 1.0, None, None, None)
    X_train = object()
    y_train = object()
    call_payload = logistic_scoring_path_call(X_train, y_train, kwargs)
    assert call_payload == (X_train, y_train, kwargs)

    with pytest.raises(ViolationError):
        logistic_scoring_fold_split(object(), np.array([1]), np.array([0]), np.array([0]))

    with pytest.raises(ViolationError):
        logistic_scoring_path_kwargs(np.array([1.0]), None, True, "lbfgs", 0, None, None, "ovr", 1e-4, 0, False, "l2", 1.0, None, None, None)

    with pytest.raises(ViolationError):
        logistic_scoring_path_kwargs(np.array([1.0]), None, True, "lbfgs", 1, None, None, "bad", 1e-4, 0, False, "l2", 1.0, None, None, None)

    with pytest.raises(ViolationError):
        logistic_scoring_classes("bad", np.array([0, 1]))

    with pytest.raises(ViolationError):
        logistic_scoring_coef_intercept_state(np.array([1.0]), "ovr", "yes")  # type: ignore[arg-type]

    with pytest.raises(ViolationError):
        logistic_scoring_score_params(np.ones((2, 2)), [], np.array([0]))  # type: ignore[arg-type]

    with pytest.raises(ViolationError):
        logistic_scoring_score_call_payload(None, object(), object(), object(), None, [])  # type: ignore[arg-type]
