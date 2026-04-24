from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.utils import check_random_state


def test_adaboost_weight_update_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.ensemble.adaboost_weight_updates import (
        adaboost_classifier_estimator_error,
        adaboost_classifier_estimator_weight,
        adaboost_classifier_sample_weight_update,
        adaboost_regressor_beta,
        adaboost_regressor_estimator_error,
        adaboost_regressor_estimator_weight,
        adaboost_regressor_loss_vector,
        adaboost_regressor_sample_weight_update,
    )

    assert callable(adaboost_classifier_estimator_error)
    assert callable(adaboost_classifier_estimator_weight)
    assert callable(adaboost_classifier_sample_weight_update)
    assert callable(adaboost_regressor_loss_vector)
    assert callable(adaboost_regressor_estimator_error)
    assert callable(adaboost_regressor_beta)
    assert callable(adaboost_regressor_estimator_weight)
    assert callable(adaboost_regressor_sample_weight_update)


def test_adaboost_classifier_weight_update_atoms_match_sklearn_boost_step() -> None:
    from sciona.atoms.ml.sklearn.ensemble.adaboost_weight_updates import (
        adaboost_classifier_estimator_error,
        adaboost_classifier_estimator_weight,
        adaboost_classifier_sample_weight_update,
    )

    X, y = make_classification(
        n_samples=60,
        n_features=6,
        n_informative=4,
        n_redundant=0,
        random_state=7,
    )
    clf = AdaBoostClassifier(n_estimators=3, learning_rate=0.7, random_state=5)
    clf._validate_estimator()
    clf.estimators_ = []
    sample_weight = np.full(X.shape[0], 1.0 / X.shape[0], dtype=np.float64)
    sklearn_sample_weight, sklearn_estimator_weight, sklearn_estimator_error = clf._boost(
        0,
        X,
        y,
        sample_weight.copy(),
        check_random_state(0),
    )

    incorrect = clf.estimators_[-1].predict(X) != y
    estimator_error = adaboost_classifier_estimator_error(incorrect, sample_weight)
    estimator_weight = adaboost_classifier_estimator_weight(
        estimator_error,
        float(clf.learning_rate),
        int(clf.n_classes_),
    )
    updated_sample_weight = adaboost_classifier_sample_weight_update(
        sample_weight,
        incorrect,
        estimator_weight,
    )

    assert np.isclose(estimator_error, sklearn_estimator_error)
    assert np.isclose(estimator_weight, sklearn_estimator_weight)
    assert np.allclose(updated_sample_weight, sklearn_sample_weight)


@pytest.mark.parametrize("loss", ["linear", "square", "exponential"])
def test_adaboost_regressor_weight_update_atoms_match_sklearn_boost_step(loss: str) -> None:
    from sciona.atoms.ml.sklearn.ensemble.adaboost_weight_updates import (
        adaboost_regressor_beta,
        adaboost_regressor_estimator_error,
        adaboost_regressor_estimator_weight,
        adaboost_regressor_loss_vector,
        adaboost_regressor_sample_weight_update,
    )

    X, y = make_regression(n_samples=70, n_features=5, noise=0.3, random_state=11)
    reg = AdaBoostRegressor(n_estimators=3, learning_rate=0.7, random_state=3, loss=loss)
    reg._validate_estimator()
    reg.estimators_ = []
    sample_weight = np.full(X.shape[0], 1.0 / X.shape[0], dtype=np.float64)
    sklearn_sample_weight, sklearn_estimator_weight, sklearn_estimator_error = reg._boost(
        0,
        X,
        y,
        sample_weight.copy(),
        check_random_state(0),
    )

    absolute_errors = np.abs(reg.estimators_[-1].predict(X) - y)
    loss_vector = adaboost_regressor_loss_vector(absolute_errors, sample_weight, loss)
    estimator_error = adaboost_regressor_estimator_error(loss_vector, sample_weight)
    beta = adaboost_regressor_beta(estimator_error)
    estimator_weight = adaboost_regressor_estimator_weight(beta, float(reg.learning_rate))
    updated_sample_weight = adaboost_regressor_sample_weight_update(
        sample_weight,
        loss_vector,
        beta,
        float(reg.learning_rate),
    )

    assert np.all((0.0 <= loss_vector) & (loss_vector <= 1.0))
    assert np.isclose(estimator_error, sklearn_estimator_error)
    assert np.isclose(estimator_weight, sklearn_estimator_weight)
    assert np.allclose(updated_sample_weight, sklearn_sample_weight)


def test_adaboost_classifier_weight_update_preserves_zero_weights() -> None:
    from sciona.atoms.ml.sklearn.ensemble.adaboost_weight_updates import adaboost_classifier_sample_weight_update

    sample_weight = np.array([0.2, 0.0, 0.3, 0.5], dtype=np.float64)
    incorrect = np.array([0, 1, 1, 0], dtype=np.int64)
    updated = adaboost_classifier_sample_weight_update(sample_weight, incorrect, 0.8)

    assert updated[1] == 0.0
    assert updated.shape == sample_weight.shape


def test_contracts_reject_invalid_adaboost_weight_update_inputs() -> None:
    from sciona.atoms.ml.sklearn.ensemble.adaboost_weight_updates import (
        adaboost_classifier_estimator_error,
        adaboost_classifier_estimator_weight,
        adaboost_regressor_beta,
        adaboost_regressor_estimator_error,
        adaboost_regressor_loss_vector,
        adaboost_regressor_sample_weight_update,
    )

    with pytest.raises(ViolationError):
        adaboost_classifier_estimator_error(
            np.array([0, 1], dtype=np.int64),
            np.array([0.2], dtype=np.float64),
        )

    with pytest.raises(ViolationError):
        adaboost_classifier_estimator_weight(0.5, 1.0, 2)

    with pytest.raises(ViolationError):
        adaboost_regressor_loss_vector(
            np.array([-1.0, 0.2], dtype=np.float64),
            np.array([0.5, 0.5], dtype=np.float64),
            "linear",
        )

    with pytest.raises(ViolationError):
        adaboost_regressor_estimator_error(
            np.array([0.2, 0.4], dtype=np.float64),
            np.array([0.4, 0.4], dtype=np.float64),
        )

    with pytest.raises(ViolationError):
        adaboost_regressor_beta(0.5)

    with pytest.raises(ViolationError):
        adaboost_regressor_sample_weight_update(
            np.array([0.5, 0.5], dtype=np.float64),
            np.array([0.2, 1.1], dtype=np.float64),
            0.4,
            1.0,
        )
