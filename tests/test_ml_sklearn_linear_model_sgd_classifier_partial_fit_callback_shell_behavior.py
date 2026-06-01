from __future__ import annotations

from types import MethodType

import numpy as np
import pytest
from icontract import ViolationError
from sklearn.linear_model import SGDClassifier


def test_sgd_classifier_partial_fit_callback_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.linear_model.sgd_classifier_partial_fit_callback_shell import (
        sgd_classifier_partial_fit_balanced_class_weight_error,
        sgd_classifier_partial_fit_callback_payload,
        sgd_classifier_partial_fit_first_call,
        sgd_classifier_partial_fit_result,
        sgd_classifier_partial_fit_validate_params_kwargs,
    )

    assert callable(sgd_classifier_partial_fit_first_call)
    assert callable(sgd_classifier_partial_fit_validate_params_kwargs)
    assert callable(sgd_classifier_partial_fit_balanced_class_weight_error)
    assert callable(sgd_classifier_partial_fit_callback_payload)
    assert callable(sgd_classifier_partial_fit_result)


def test_sgd_classifier_partial_fit_callback_payload_matches_sklearn_delegation() -> None:
    from sciona.atoms.ml.sklearn.linear_model.sgd_classifier_partial_fit_callback_shell import (
        sgd_classifier_partial_fit_callback_payload,
        sgd_classifier_partial_fit_validate_params_kwargs,
    )

    clf = SGDClassifier(alpha=0.002, loss="log_loss", learning_rate="constant", eta0=0.01)
    X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
    y = np.array([0, 1])
    classes = np.array([0, 1])
    sample_weight = np.array([1.0, 2.0], dtype=np.float64)
    captured: dict[str, object] = {}

    def fake_more_validate_params(self: SGDClassifier, **kwargs: object) -> None:
        captured["validated"] = self
        captured["validate_kwargs"] = kwargs
        return None

    def fake_partial_fit(self: SGDClassifier, X_arg: object, y_arg: object, **kwargs: object) -> SGDClassifier:
        captured["X"] = X_arg
        captured["y"] = y_arg
        captured.update(kwargs)
        return self

    clf._more_validate_params = MethodType(fake_more_validate_params, clf)
    clf._partial_fit = MethodType(fake_partial_fit, clf)

    result = clf.partial_fit(X, y, classes=classes, sample_weight=sample_weight)
    payload = sgd_classifier_partial_fit_callback_payload(
        X,
        y,
        alpha=clf.alpha,
        loss=clf.loss,
        learning_rate=clf.learning_rate,
        classes=classes,
        sample_weight=sample_weight,
    )

    assert result is clf
    assert captured["validated"] is clf
    assert captured["validate_kwargs"] == sgd_classifier_partial_fit_validate_params_kwargs(True)
    for key, value in payload.items():
        if key in {"X", "y", "classes", "sample_weight", "coef_init", "intercept_init"}:
            assert captured[key] is value
        else:
            assert captured[key] == value


def test_sgd_classifier_partial_fit_first_call_and_balanced_message() -> None:
    from sciona.atoms.ml.sklearn.linear_model.sgd_classifier_partial_fit_callback_shell import (
        sgd_classifier_partial_fit_balanced_class_weight_error,
        sgd_classifier_partial_fit_first_call,
        sgd_classifier_partial_fit_validate_params_kwargs,
    )

    expected = (
        "class_weight 'balanced' is not supported for "
        "partial_fit. In order to use 'balanced' weights,"
        " use compute_class_weight('balanced', "
        "classes=classes, y=y). "
        "In place of y you can use a large enough sample "
        "of the full training set target to properly "
        "estimate the class frequency distributions. "
        "Pass the resulting weights as the class_weight "
        "parameter."
    )

    assert sgd_classifier_partial_fit_first_call(False) is True
    assert sgd_classifier_partial_fit_first_call(True) is False
    assert sgd_classifier_partial_fit_validate_params_kwargs(True) == {"for_partial_fit": True}
    assert sgd_classifier_partial_fit_validate_params_kwargs(False) == {}
    assert sgd_classifier_partial_fit_balanced_class_weight_error("balanced") == expected


def test_sgd_classifier_partial_fit_payload_preserves_optional_objects() -> None:
    from sciona.atoms.ml.sklearn.linear_model.sgd_classifier_partial_fit_callback_shell import (
        sgd_classifier_partial_fit_callback_payload,
        sgd_classifier_partial_fit_result,
    )

    X = object()
    y = object()
    classes = object()
    sample_weight = object()
    delegated_result = object()

    payload = sgd_classifier_partial_fit_callback_payload(
        X,
        y,
        alpha=0.5,
        loss="hinge",
        learning_rate="optimal",
        classes=classes,
        sample_weight=sample_weight,
    )

    assert payload["X"] is X
    assert payload["y"] is y
    assert payload["alpha"] == 0.5
    assert payload["C"] == 1.0
    assert payload["loss"] == "hinge"
    assert payload["learning_rate"] == "optimal"
    assert payload["max_iter"] == 1
    assert payload["classes"] is classes
    assert payload["sample_weight"] is sample_weight
    assert payload["coef_init"] is None
    assert payload["intercept_init"] is None
    assert sgd_classifier_partial_fit_result(delegated_result) is delegated_result


def test_sgd_classifier_partial_fit_callback_shell_contracts() -> None:
    from sciona.atoms.ml.sklearn.linear_model.sgd_classifier_partial_fit_callback_shell import (
        sgd_classifier_partial_fit_balanced_class_weight_error,
        sgd_classifier_partial_fit_callback_payload,
        sgd_classifier_partial_fit_first_call,
        sgd_classifier_partial_fit_result,
    )

    with pytest.raises(ViolationError):
        sgd_classifier_partial_fit_first_call(1)

    with pytest.raises(ViolationError):
        sgd_classifier_partial_fit_balanced_class_weight_error("auto")

    with pytest.raises(ViolationError):
        sgd_classifier_partial_fit_callback_payload(None, object(), alpha=0.1, loss="hinge", learning_rate="optimal", classes=object())

    with pytest.raises(ViolationError):
        sgd_classifier_partial_fit_callback_payload(object(), object(), alpha=float("nan"), loss="hinge", learning_rate="optimal", classes=object())

    with pytest.raises(ViolationError):
        sgd_classifier_partial_fit_callback_payload(object(), object(), alpha=0.1, loss="", learning_rate="optimal", classes=object())

    with pytest.raises(ViolationError):
        sgd_classifier_partial_fit_result(None)
