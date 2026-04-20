from __future__ import annotations

import numpy as np
import pytest
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as SklearnQDA


def _data() -> tuple[np.ndarray, np.ndarray]:
    class_zero = np.array(
        [
            [0.0, 0.0],
            [0.2, 0.4],
            [0.5, -0.1],
            [0.8, 0.3],
            [1.0, 0.7],
        ],
        dtype=np.float64,
    )
    class_one = np.array(
        [
            [3.0, 3.5],
            [3.4, 4.1],
            [3.8, 3.2],
            [4.2, 3.9],
            [4.5, 4.4],
        ],
        dtype=np.float64,
    )
    class_two = np.array(
        [
            [-3.0, 2.5],
            [-2.7, 2.9],
            [-2.3, 2.2],
            [-1.9, 2.8],
            [-1.6, 3.3],
        ],
        dtype=np.float64,
    )
    X = np.vstack([class_zero, class_one, class_two])
    y = np.repeat(np.array([0.0, 1.0, 2.0], dtype=np.float64), 5)
    return X, y


def test_qda_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.discriminant_analysis import (
        QDAState,
        qda_decision_function,
        qda_fit,
        qda_predict,
        qda_predict_log_proba,
        qda_predict_proba,
    )

    assert QDAState is not None
    assert callable(qda_fit)
    assert callable(qda_decision_function)
    assert callable(qda_predict_log_proba)
    assert callable(qda_predict_proba)
    assert callable(qda_predict)


def test_qda_fit_and_predictions_match_sklearn() -> None:
    from sciona.atoms.ml.sklearn.discriminant_analysis import (
        qda_decision_function,
        qda_fit,
        qda_predict,
        qda_predict_log_proba,
        qda_predict_proba,
    )

    X, y = _data()
    state = qda_fit(X, y, store_covariance=True)
    expected = SklearnQDA(store_covariance=True).fit(X, y)

    assert np.allclose(state.classes, expected.classes_)
    assert np.allclose(state.priors, expected.priors_)
    assert np.allclose(state.means, expected.means_)
    for result, sklearn_result in zip(state.scalings, expected.scalings_):
        assert np.allclose(result, sklearn_result)
    for result, sklearn_result in zip(state.rotations, expected.rotations_):
        assert np.allclose(np.abs(result), np.abs(sklearn_result))
    assert state.covariance is not None
    for result, sklearn_result in zip(state.covariance, expected.covariance_):
        assert np.allclose(result, sklearn_result)

    query = X[[0, 4, 7, 12]]
    assert np.allclose(qda_decision_function(query, state), expected._decision_function(query))
    assert np.allclose(qda_predict_log_proba(query, state), expected.predict_log_proba(query))
    assert np.allclose(qda_predict_proba(query, state), expected.predict_proba(query))
    assert np.allclose(qda_predict(query, state), expected.predict(query))


def test_qda_fit_with_priors_and_regularization_matches_sklearn() -> None:
    from sciona.atoms.ml.sklearn.discriminant_analysis import qda_fit, qda_predict_proba

    X, y = _data()
    priors = (0.2, 0.5, 0.3)
    state = qda_fit(X, y, priors=priors, reg_param=0.1)
    expected = SklearnQDA(priors=np.asarray(priors), reg_param=0.1).fit(X, y)

    assert state.covariance is None
    assert np.allclose(state.priors, expected.priors_)
    assert np.allclose(qda_predict_proba(X[:5], state), expected.predict_proba(X[:5]))


def test_qda_rejects_rank_and_prior_errors() -> None:
    from sciona.atoms.ml.sklearn.discriminant_analysis import qda_fit

    X, y = _data()
    with pytest.raises(Exception):
        qda_fit(X, y, priors=(0.5, 0.5))

    low_rank_x = np.array([[0.0, 0.0], [0.2, 0.2], [2.0, 2.0], [2.2, 2.2]], dtype=np.float64)
    low_rank_y = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float64)
    with pytest.raises(Exception):
        qda_fit(low_rank_x, low_rank_y)
