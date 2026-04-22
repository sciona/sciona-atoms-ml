from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError
from sklearn.linear_model import SGDClassifier


def test_sgd_helper_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.linear_model.sgd import (
        passive_aggressive_classifier_sgd_config,
        passive_aggressive_regressor_sgd_config,
        sgd_l1_ratio_or_zero,
        sgd_learning_rate_value,
        sgd_modified_huber_proba,
        sgd_passive_aggressive_step_size,
    )

    assert callable(sgd_l1_ratio_or_zero)
    assert callable(sgd_learning_rate_value)
    assert callable(sgd_passive_aggressive_step_size)
    assert callable(sgd_modified_huber_proba)
    assert callable(passive_aggressive_classifier_sgd_config)
    assert callable(passive_aggressive_regressor_sgd_config)


def test_sgd_l1_ratio_or_zero_matches_base_sgd_helper() -> None:
    from sciona.atoms.ml.sklearn.linear_model.sgd import sgd_l1_ratio_or_zero

    assert sgd_l1_ratio_or_zero(None) == 0.0
    assert sgd_l1_ratio_or_zero(0.25) == pytest.approx(0.25)


def test_sgd_learning_rate_value_matches_documented_schedules() -> None:
    from sciona.atoms.ml.sklearn.linear_model.sgd import sgd_learning_rate_value

    assert sgd_learning_rate_value("constant", 0.2, t=3.0) == pytest.approx(0.2)
    assert sgd_learning_rate_value("adaptive", 0.2, t=3.0) == pytest.approx(0.2)
    assert sgd_learning_rate_value("invscaling", 0.2, t=4.0, power_t=0.5) == pytest.approx(0.1)
    assert sgd_learning_rate_value("optimal", 0.2, alpha=0.01, t=4.0, t0=6.0) == pytest.approx(10.0)


def test_sgd_passive_aggressive_step_sizes_match_sklearn_docs() -> None:
    from sciona.atoms.ml.sklearn.linear_model.sgd import sgd_passive_aggressive_step_size

    assert sgd_passive_aggressive_step_size(3.0, 2.0, 1.0, learning_rate="pa1") == pytest.approx(1.0)
    assert sgd_passive_aggressive_step_size(0.5, 2.0, 1.0, learning_rate="pa1") == pytest.approx(0.25)
    assert sgd_passive_aggressive_step_size(3.0, 2.0, 1.0, learning_rate="pa2") == pytest.approx(1.2)


def test_sgd_modified_huber_binary_proba_matches_sklearn() -> None:
    from sciona.atoms.ml.sklearn.linear_model.sgd import sgd_modified_huber_proba

    scores = np.array([-2.0, -0.5, 0.0, 0.5, 2.0], dtype=np.float64)
    atom_proba = sgd_modified_huber_proba(scores, binary=True)

    clf = SGDClassifier(loss="modified_huber")
    clf.classes_ = np.array([0, 1])
    clf.coef_ = np.array([[1.0]], dtype=np.float64)
    clf.intercept_ = np.array([0.0], dtype=np.float64)
    clf.n_features_in_ = 1
    sklearn_proba = clf.predict_proba(scores.reshape(-1, 1))

    assert np.allclose(atom_proba, sklearn_proba)


def test_sgd_modified_huber_multiclass_proba_matches_sklearn() -> None:
    from sciona.atoms.ml.sklearn.linear_model.sgd import sgd_modified_huber_proba

    scores = np.array(
        [
            [-2.0, -2.0, -2.0],
            [2.0, 0.0, -2.0],
            [-0.5, 0.5, 2.0],
        ],
        dtype=np.float64,
    )
    atom_proba = sgd_modified_huber_proba(scores, binary=False)

    clf = SGDClassifier(loss="modified_huber")
    clf.classes_ = np.array([0, 1, 2])
    clf.coef_ = np.eye(3, dtype=np.float64)
    clf.intercept_ = np.zeros(3, dtype=np.float64)
    clf.n_features_in_ = 3
    sklearn_proba = clf.predict_proba(scores)

    assert np.allclose(atom_proba, sklearn_proba)
    assert np.allclose(atom_proba[0], np.full(3, 1.0 / 3.0))


def test_passive_aggressive_configs_match_source_delegation() -> None:
    from sciona.atoms.ml.sklearn.linear_model.sgd import (
        passive_aggressive_classifier_sgd_config,
        passive_aggressive_regressor_sgd_config,
    )

    assert passive_aggressive_classifier_sgd_config("hinge") == ("hinge", "pa1", 1.0)
    assert passive_aggressive_classifier_sgd_config("squared_hinge") == ("hinge", "pa2", 1.0)
    assert passive_aggressive_regressor_sgd_config("epsilon_insensitive") == ("epsilon_insensitive", "pa1", 1.0)
    assert passive_aggressive_regressor_sgd_config("squared_epsilon_insensitive") == ("epsilon_insensitive", "pa2", 1.0)


def test_contracts_reject_invalid_sgd_helper_inputs() -> None:
    from sciona.atoms.ml.sklearn.linear_model.sgd import (
        passive_aggressive_classifier_sgd_config,
        sgd_l1_ratio_or_zero,
        sgd_learning_rate_value,
        sgd_modified_huber_proba,
        sgd_passive_aggressive_step_size,
    )

    with pytest.raises(ViolationError):
        sgd_l1_ratio_or_zero(2.0)

    with pytest.raises(ViolationError):
        sgd_learning_rate_value("optimal", 0.1, alpha=0.0)

    with pytest.raises(ViolationError):
        sgd_passive_aggressive_step_size(1.0, 0.0, 1.0, learning_rate="pa1")

    with pytest.raises(ViolationError):
        sgd_modified_huber_proba(np.ones((2, 2), dtype=np.float64), binary=True)

    with pytest.raises(ViolationError):
        passive_aggressive_classifier_sgd_config("log_loss")
