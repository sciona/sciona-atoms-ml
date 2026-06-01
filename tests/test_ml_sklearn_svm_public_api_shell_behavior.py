from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError
from sklearn.svm import LinearSVC, LinearSVR, NuSVC, NuSVR, OneClassSVM, SVC, SVR


def _classification_data() -> tuple[np.ndarray, np.ndarray]:
    X = np.array(
        [[-2.0, -1.0], [-1.0, -2.0], [-1.5, -1.2], [1.0, 1.0], [1.5, 1.2], [2.0, 1.0]],
        dtype=np.float64,
    )
    y = np.array([0, 0, 0, 1, 1, 1])
    return X, y


def _regression_data() -> tuple[np.ndarray, np.ndarray]:
    X = np.array([[0.0, 0.0], [0.5, 0.2], [1.0, 0.8], [1.5, 1.0], [2.0, 1.8], [2.5, 2.0]], dtype=np.float64)
    y = np.array([0.0, 0.3, 0.9, 1.4, 2.1, 2.4], dtype=np.float64)
    return X, y


def test_svm_public_api_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.svm.public_api_shell import (
        svm_estimator_backend,
        svm_estimator_catalog,
        svm_estimator_methods,
        svm_estimator_task,
        svm_fit_return_self,
        svm_liblinear_fitted_state,
        svm_libsvm_fitted_support_state,
        svm_linear_fit_liblinear_payload,
        svm_prediction_method_payload,
    )

    assert callable(svm_estimator_backend)
    assert callable(svm_estimator_catalog)
    assert callable(svm_estimator_methods)
    assert callable(svm_estimator_task)
    assert callable(svm_fit_return_self)
    assert callable(svm_liblinear_fitted_state)
    assert callable(svm_libsvm_fitted_support_state)
    assert callable(svm_linear_fit_liblinear_payload)
    assert callable(svm_prediction_method_payload)


def test_svm_public_catalog_and_capabilities() -> None:
    from sciona.atoms.ml.sklearn.svm.public_api_shell import (
        svm_estimator_backend,
        svm_estimator_catalog,
        svm_estimator_methods,
        svm_estimator_task,
    )

    assert svm_estimator_catalog() == ("SVC", "NuSVC", "SVR", "NuSVR", "LinearSVC", "LinearSVR", "OneClassSVM")
    assert svm_estimator_backend("SVC") == "libsvm"
    assert svm_estimator_backend("LinearSVC") == "liblinear"
    assert svm_estimator_task("NuSVC") == "classification"
    assert svm_estimator_task("LinearSVR") == "regression"
    assert svm_estimator_task("OneClassSVM") == "outlier_detection"
    assert "predict_proba" not in svm_estimator_methods("SVC")
    assert "predict_proba" in svm_estimator_methods("SVC", probability_enabled=True)
    assert svm_estimator_methods("OneClassSVM") == ("fit", "fit_predict", "predict", "decision_function", "score_samples")


def test_svm_linear_fit_liblinear_payload_matches_source_arguments() -> None:
    from sciona.atoms.ml.sklearn.svm.public_api_shell import svm_linear_fit_liblinear_payload

    sample_weight = np.ones(6, dtype=np.float64)
    svc_payload = svm_linear_fit_liblinear_payload(
        "LinearSVC",
        2.0,
        True,
        1.5,
        "balanced",
        "l2",
        False,
        0,
        500,
        1e-4,
        13,
        "ovr",
        "squared_hinge",
        sample_weight=sample_weight,
    )
    assert svc_payload["backend"] == "liblinear"
    assert svc_payload["class_weight"] == "balanced"
    assert svc_payload["sample_weight"] is sample_weight
    assert svc_payload["loss"] == "squared_hinge"

    svr_payload = svm_linear_fit_liblinear_payload(
        "LinearSVR",
        1.0,
        False,
        1.0,
        {"ignored": 2.0},
        "l1",
        True,
        0,
        100,
        1e-5,
        None,
        "crammer_singer",
        "epsilon_insensitive",
        epsilon=0.2,
    )
    assert svr_payload["class_weight"] is None
    assert svr_payload["penalty"] == "l2"
    assert svr_payload["multi_class"] == "ovr"
    assert svr_payload["epsilon"] == 0.2


def test_svm_public_shell_matches_fitted_sklearn_objects() -> None:
    from sciona.atoms.ml.sklearn.svm.public_api_shell import (
        svm_fit_return_self,
        svm_liblinear_fitted_state,
        svm_libsvm_fitted_support_state,
        svm_prediction_method_payload,
    )

    Xc, yc = _classification_data()
    Xr, yr = _regression_data()
    Xq = Xc[:3]

    estimators = [
        SVC(kernel="linear", probability=True, gamma="auto", random_state=0).fit(Xc, yc),
        NuSVC(kernel="linear", nu=0.3, gamma="auto", probability=True, random_state=0).fit(Xc, yc),
        SVR(kernel="linear", C=1.0).fit(Xr, yr),
        NuSVR(kernel="linear", nu=0.4, C=1.0).fit(Xr, yr),
        OneClassSVM(kernel="linear", nu=0.4).fit(Xc),
    ]
    for estimator in estimators:
        assert svm_fit_return_self(estimator) is estimator
        state = svm_libsvm_fitted_support_state(estimator)
        assert state["backend"] == "libsvm"
        assert state["estimator_name"] == estimator.__class__.__name__
        assert np.array_equal(state["support"], estimator.support_)
        payload = svm_prediction_method_payload(estimator, "predict", Xq)
        assert payload["estimator"] is estimator
        assert payload["method_name"] == "predict"
        assert payload["args"] == (Xq,)
        assert np.array_equal(getattr(payload["estimator"], payload["method_name"])(*payload["args"]), estimator.predict(Xq))

    linear_clf = LinearSVC(random_state=0, dual=False, max_iter=2000).fit(Xc, yc)
    linear_reg = LinearSVR(random_state=0, dual=True, max_iter=2000).fit(Xr, yr)
    for estimator, X in [(linear_clf, Xc), (linear_reg, Xr)]:
        assert svm_fit_return_self(estimator) is estimator
        state = svm_liblinear_fitted_state(estimator)
        assert state["backend"] == "liblinear"
        assert state["estimator_name"] == estimator.__class__.__name__
        assert np.array_equal(state["coef"], estimator.coef_)
        assert np.array_equal(state["intercept"], estimator.intercept_)
        payload = svm_prediction_method_payload(estimator, "predict", X[:2])
        assert np.array_equal(getattr(payload["estimator"], payload["method_name"])(*payload["args"]), estimator.predict(X[:2]))


def test_svm_public_shell_contracts() -> None:
    from sciona.atoms.ml.sklearn.svm.public_api_shell import (
        svm_estimator_backend,
        svm_fit_return_self,
        svm_linear_fit_liblinear_payload,
        svm_prediction_method_payload,
    )

    with pytest.raises(ViolationError):
        svm_estimator_backend("UnknownSVM")

    with pytest.raises(ViolationError):
        svm_linear_fit_liblinear_payload("SVC", 1.0, True, 1.0, None, "l2", True, 0, 100, 1e-4, None, "ovr", "hinge")

    with pytest.raises(ViolationError):
        svm_fit_return_self(SVC())

    fitted = SVC(probability=False).fit(*_classification_data())
    with pytest.raises(ViolationError):
        svm_prediction_method_payload(fitted, "predict_proba", _classification_data()[0])
