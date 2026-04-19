from __future__ import annotations

import numpy as np
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.calibration import _SigmoidCalibration as SklearnSigmoidCalibration
from sklearn.calibration import _TemperatureScaling as SklearnTemperatureScaling
from sklearn.calibration import calibration_curve as sklearn_calibration_curve
from sklearn.datasets import make_classification
from sklearn.naive_bayes import GaussianNB


def test_calibration_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.calibration import (
        calibrated_classifier_cv_fit,
        calibrated_classifier_cv_predict,
        calibrated_classifier_cv_predict_proba,
        calibration_curve,
        sigmoid_calibration_fit,
        sigmoid_calibration_predict,
        temperature_scaling_fit,
        temperature_scaling_predict,
    )

    assert callable(calibrated_classifier_cv_fit)
    assert callable(calibrated_classifier_cv_predict)
    assert callable(calibrated_classifier_cv_predict_proba)
    assert callable(calibration_curve)
    assert callable(sigmoid_calibration_fit)
    assert callable(sigmoid_calibration_predict)
    assert callable(temperature_scaling_fit)
    assert callable(temperature_scaling_predict)


def test_calibration_curve_matches_sklearn_uniform() -> None:
    from sciona.atoms.ml.sklearn.calibration import calibration_curve

    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.3, 0.4, 0.65, 0.7, 0.8, 0.9, 1.0])
    result = calibration_curve(y_true, y_prob, n_bins=3)
    expected = sklearn_calibration_curve(y_true, y_prob, n_bins=3)
    assert np.allclose(result[0], expected[0])
    assert np.allclose(result[1], expected[1])


def test_calibration_curve_matches_sklearn_quantile() -> None:
    from sciona.atoms.ml.sklearn.calibration import calibration_curve

    y_true = np.array([False, False, True, False, True, True, True, False])
    y_prob = np.array([0.01, 0.15, 0.2, 0.35, 0.55, 0.7, 0.8, 0.95])
    result = calibration_curve(y_true, y_prob, n_bins=4, strategy="quantile")
    expected = sklearn_calibration_curve(y_true, y_prob, n_bins=4, strategy="quantile")
    assert np.allclose(result[0], expected[0])
    assert np.allclose(result[1], expected[1])


def test_calibration_curve_matches_sklearn_string_pos_label() -> None:
    from sciona.atoms.ml.sklearn.calibration import calibration_curve

    y_true = np.array(["no", "yes", "no", "yes", "yes", "no"])
    y_prob = np.array([0.05, 0.2, 0.4, 0.8, 0.9, 0.6])
    result = calibration_curve(y_true, y_prob, pos_label="yes", n_bins=3)
    expected = sklearn_calibration_curve(y_true, y_prob, pos_label="yes", n_bins=3)
    assert np.allclose(result[0], expected[0])
    assert np.allclose(result[1], expected[1])


def test_calibration_curve_omits_empty_bins_like_sklearn() -> None:
    from sciona.atoms.ml.sklearn.calibration import calibration_curve

    y_true = np.array([0, 1, 1])
    y_prob = np.array([0.05, 0.9, 0.95])
    result = calibration_curve(y_true, y_prob, n_bins=10)
    expected = sklearn_calibration_curve(y_true, y_prob, n_bins=10)
    assert result[0].shape[0] < 10
    assert np.allclose(result[0], expected[0])
    assert np.allclose(result[1], expected[1])


def test_calibration_curve_multiclass_error_matches_sklearn() -> None:
    from sciona.atoms.ml.sklearn.calibration import calibration_curve

    y_true = np.array([0, 1, 2])
    y_prob = np.array([0.1, 0.5, 0.9])
    with pytest.raises(ValueError, match="Only binary classification"):
        calibration_curve(y_true, y_prob, pos_label=1)
    with pytest.raises(ValueError, match="Only binary classification"):
        sklearn_calibration_curve(y_true, y_prob, pos_label=1)


def test_calibration_curve_rejects_bad_probability_range() -> None:
    from sciona.atoms.ml.sklearn.calibration import calibration_curve

    with pytest.raises(Exception):
        calibration_curve(np.array([0, 1]), np.array([0.2, 1.2]))


def test_sigmoid_calibration_matches_sklearn_private_calibrator() -> None:
    from sciona.atoms.ml.sklearn.calibration import sigmoid_calibration_fit, sigmoid_calibration_predict

    predictions = np.array([-2.0, -1.0, -0.2, 0.4, 1.0, 2.0], dtype=np.float64)
    y = np.array([0, 0, 0, 1, 1, 1], dtype=np.float64)
    state = sigmoid_calibration_fit(predictions, y)
    expected = SklearnSigmoidCalibration().fit(predictions, y)

    query = np.array([-1.5, 0.0, 1.5], dtype=np.float64)
    assert np.isclose(state.a, expected.a_)
    assert np.isclose(state.b, expected.b_)
    assert np.allclose(sigmoid_calibration_predict(query, state), expected.predict(query))


def test_temperature_scaling_matches_sklearn_private_calibrator() -> None:
    from sciona.atoms.ml.sklearn.calibration import temperature_scaling_fit, temperature_scaling_predict

    scores = np.array([[-1.0, 1.0], [-0.4, 0.4], [0.8, -0.8], [1.5, -1.5]], dtype=np.float64)
    y = np.array([1, 1, 0, 0], dtype=np.float64)
    state = temperature_scaling_fit(scores, y)
    expected = SklearnTemperatureScaling().fit(scores, y)

    query = np.array([[-0.6, 0.6], [1.2, -1.2]], dtype=np.float64)
    assert np.isclose(state.beta, expected.beta_)
    assert np.allclose(temperature_scaling_predict(query, state), expected.predict(query))


def test_calibrated_classifier_cv_fit_predict_matches_sklearn() -> None:
    from sciona.atoms.ml.sklearn.calibration import (
        calibrated_classifier_cv_fit,
        calibrated_classifier_cv_predict,
        calibrated_classifier_cv_predict_proba,
    )

    X, y = make_classification(
        n_samples=80,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        random_state=0,
    )
    query = X[:5]
    estimator = GaussianNB()

    state = calibrated_classifier_cv_fit(estimator, X, y, method="sigmoid", cv=3, ensemble=False)
    expected = CalibratedClassifierCV(estimator=GaussianNB(), method="sigmoid", cv=3, ensemble=False).fit(X, y)

    assert np.array_equal(state.classes.astype(int), expected.classes_)
    assert np.allclose(calibrated_classifier_cv_predict_proba(query, state), expected.predict_proba(query))
    assert np.array_equal(calibrated_classifier_cv_predict(query, state).astype(int), expected.predict(query))
