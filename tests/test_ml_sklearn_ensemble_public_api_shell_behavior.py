from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    IsolationForest,
    RandomForestClassifier,
    RandomForestRegressor,
)


def _classification_data() -> tuple[np.ndarray, np.ndarray]:
    X = np.array(
        [[-2.0, -1.0], [-1.0, -2.0], [-1.5, -1.2], [1.0, 1.0], [1.5, 1.2], [2.0, 1.0], [1.8, 0.7], [-1.7, -0.8]],
        dtype=np.float64,
    )
    y = np.array([0, 0, 0, 1, 1, 1, 1, 0])
    return X, y


def _regression_data() -> tuple[np.ndarray, np.ndarray]:
    X = np.array([[0.0, 0.0], [0.5, 0.2], [1.0, 0.8], [1.5, 1.0], [2.0, 1.8], [2.5, 2.0], [3.0, 2.6]], dtype=np.float64)
    y = np.array([0.0, 0.3, 0.9, 1.4, 2.1, 2.4, 3.0], dtype=np.float64)
    return X, y


def test_ensemble_public_api_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.ensemble.public_api_shell import (
        ensemble_estimator_backend,
        ensemble_estimator_catalog,
        ensemble_estimator_family,
        ensemble_estimator_methods,
        ensemble_estimator_task,
        ensemble_fit_return_self,
        ensemble_fitted_state_summary,
        ensemble_prediction_method_payload,
    )

    assert callable(ensemble_estimator_backend)
    assert callable(ensemble_estimator_catalog)
    assert callable(ensemble_estimator_family)
    assert callable(ensemble_estimator_methods)
    assert callable(ensemble_estimator_task)
    assert callable(ensemble_fit_return_self)
    assert callable(ensemble_fitted_state_summary)
    assert callable(ensemble_prediction_method_payload)


def test_ensemble_public_catalog_and_capabilities() -> None:
    from sciona.atoms.ml.sklearn.ensemble.public_api_shell import (
        ensemble_estimator_backend,
        ensemble_estimator_catalog,
        ensemble_estimator_family,
        ensemble_estimator_methods,
        ensemble_estimator_task,
    )

    assert ensemble_estimator_catalog() == (
        "RandomForestClassifier",
        "RandomForestRegressor",
        "ExtraTreesClassifier",
        "ExtraTreesRegressor",
        "GradientBoostingClassifier",
        "GradientBoostingRegressor",
        "HistGradientBoostingClassifier",
        "HistGradientBoostingRegressor",
        "IsolationForest",
    )
    assert ensemble_estimator_family("RandomForestClassifier") == "random_forest"
    assert ensemble_estimator_family("ExtraTreesRegressor") == "extra_trees"
    assert ensemble_estimator_family("HistGradientBoostingClassifier") == "hist_gradient_boosting"
    assert ensemble_estimator_task("GradientBoostingRegressor") == "regression"
    assert ensemble_estimator_task("IsolationForest") == "outlier_detection"
    assert ensemble_estimator_backend("ExtraTreesClassifier") == "cython_tree_ensemble"
    assert ensemble_estimator_backend("GradientBoostingClassifier") == "python_tree_boosting"
    assert ensemble_estimator_backend("HistGradientBoostingRegressor") == "histogram_boosting"
    assert "predict_proba" in ensemble_estimator_methods("RandomForestClassifier")
    assert "apply" in ensemble_estimator_methods("GradientBoostingClassifier")
    assert "predict_log_proba" not in ensemble_estimator_methods("HistGradientBoostingClassifier")
    assert ensemble_estimator_methods("IsolationForest") == ("fit", "fit_predict", "predict", "decision_function", "score_samples")


def test_ensemble_public_shell_matches_fitted_sklearn_objects() -> None:
    from sciona.atoms.ml.sklearn.ensemble.public_api_shell import (
        ensemble_fit_return_self,
        ensemble_fitted_state_summary,
        ensemble_prediction_method_payload,
    )

    Xc, yc = _classification_data()
    Xr, yr = _regression_data()
    Xq = Xc[:3]

    estimators = [
        RandomForestClassifier(n_estimators=3, random_state=0, max_depth=2).fit(Xc, yc),
        ExtraTreesClassifier(n_estimators=3, random_state=0, max_depth=2).fit(Xc, yc),
        GradientBoostingClassifier(n_estimators=3, random_state=0, max_depth=1).fit(Xc, yc),
        HistGradientBoostingClassifier(max_iter=3, random_state=0, min_samples_leaf=1).fit(Xc, yc),
        RandomForestRegressor(n_estimators=3, random_state=0, max_depth=2).fit(Xr, yr),
        ExtraTreesRegressor(n_estimators=3, random_state=0, max_depth=2).fit(Xr, yr),
        GradientBoostingRegressor(n_estimators=3, random_state=0, max_depth=1).fit(Xr, yr),
        HistGradientBoostingRegressor(max_iter=3, random_state=0, min_samples_leaf=1).fit(Xr, yr),
        IsolationForest(n_estimators=3, random_state=0, max_samples=4).fit(Xc),
    ]
    for estimator in estimators:
        assert ensemble_fit_return_self(estimator) is estimator
        summary = ensemble_fitted_state_summary(estimator)
        assert summary["estimator_name"] == estimator.__class__.__name__
        assert summary["n_features_in"] == estimator.n_features_in_
        assert summary["estimator_count"] >= 1

        query = Xq if summary["task"] != "regression" else Xr[:3]
        payload = ensemble_prediction_method_payload(estimator, "predict", query)
        assert payload["estimator"] is estimator
        assert payload["method_name"] == "predict"
        assert payload["args"] == (query,)
        assert np.asarray(getattr(payload["estimator"], payload["method_name"])(*payload["args"])).shape[0] == query.shape[0]


def test_ensemble_public_shell_contracts() -> None:
    from sciona.atoms.ml.sklearn.ensemble.public_api_shell import (
        ensemble_estimator_backend,
        ensemble_estimator_catalog,
        ensemble_fit_return_self,
        ensemble_prediction_method_payload,
    )

    with pytest.raises(ViolationError):
        ensemble_estimator_catalog("all_estimators")

    with pytest.raises(ViolationError):
        ensemble_estimator_backend("BaggingClassifier")

    with pytest.raises(ViolationError):
        ensemble_fit_return_self(RandomForestClassifier(n_estimators=1))

    fitted = RandomForestClassifier(n_estimators=2, random_state=0).fit(*_classification_data())
    with pytest.raises(ViolationError):
        ensemble_prediction_method_payload(fitted, "score_samples", _classification_data()[0])
