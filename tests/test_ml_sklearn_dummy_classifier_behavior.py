from __future__ import annotations

import numpy as np
import pytest
from sklearn.dummy import DummyClassifier as SklearnDummyClassifier


def test_dummy_classifier_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.dummy import (
        DummyClassifierState,
        dummy_classifier_fit,
        dummy_classifier_predict,
        dummy_classifier_predict_proba,
    )

    assert DummyClassifierState is not None
    assert callable(dummy_classifier_fit)
    assert callable(dummy_classifier_predict)
    assert callable(dummy_classifier_predict_proba)


def _assert_single_output_matches(state, expected: SklearnDummyClassifier) -> None:
    assert state.n_outputs == 1
    assert np.allclose(state.classes[0], expected.classes_)
    assert np.allclose(state.class_prior[0], expected.class_prior_)
    assert state.n_classes[0] == expected.n_classes_


def test_dummy_classifier_prior_predict_and_proba_match_sklearn() -> None:
    from sciona.atoms.ml.sklearn.dummy import dummy_classifier_fit, dummy_classifier_predict, dummy_classifier_predict_proba

    X = np.arange(12, dtype=np.float64).reshape(6, 2)
    y = np.array([0.0, 1.0, 1.0, 2.0, 2.0, 2.0], dtype=np.float64)
    state = dummy_classifier_fit(y, strategy="prior")
    expected = SklearnDummyClassifier(strategy="prior").fit(X, y)

    _assert_single_output_matches(state, expected)
    assert np.allclose(dummy_classifier_predict(X[:3], state), expected.predict(X[:3]))
    assert np.allclose(dummy_classifier_predict_proba(X[:3], state), expected.predict_proba(X[:3]))


def test_dummy_classifier_most_frequent_matches_sklearn_multioutput() -> None:
    from sciona.atoms.ml.sklearn.dummy import dummy_classifier_fit, dummy_classifier_predict, dummy_classifier_predict_proba

    X = np.arange(12, dtype=np.float64).reshape(6, 2)
    y = np.array(
        [
            [0.0, 1.0],
            [1.0, 1.0],
            [1.0, 2.0],
            [2.0, 2.0],
            [2.0, 2.0],
            [2.0, 3.0],
        ],
        dtype=np.float64,
    )
    state = dummy_classifier_fit(y, strategy="most_frequent")
    expected = SklearnDummyClassifier(strategy="most_frequent").fit(X, y)

    assert state.n_outputs == expected.n_outputs_
    assert np.allclose(dummy_classifier_predict(X[:3], state), expected.predict(X[:3]))
    result_proba = dummy_classifier_predict_proba(X[:3], state)
    assert isinstance(result_proba, tuple)
    expected_proba = expected.predict_proba(X[:3])
    assert len(result_proba) == len(expected_proba)
    for result, expected_array in zip(result_proba, expected_proba):
        assert np.allclose(result, expected_array)


def test_dummy_classifier_constant_matches_sklearn() -> None:
    from sciona.atoms.ml.sklearn.dummy import dummy_classifier_fit, dummy_classifier_predict, dummy_classifier_predict_proba

    X = np.arange(8, dtype=np.float64).reshape(4, 2)
    y = np.array([[1.0, 2.0], [3.0, 4.0], [3.0, 2.0], [1.0, 4.0]], dtype=np.float64)
    state = dummy_classifier_fit(y, strategy="constant", constant=(3.0, 4.0))
    expected = SklearnDummyClassifier(strategy="constant", constant=np.array([3.0, 4.0])).fit(X, y)

    assert np.allclose(dummy_classifier_predict(X[:2], state), expected.predict(X[:2]))
    result_proba = dummy_classifier_predict_proba(X[:2], state)
    expected_proba = expected.predict_proba(X[:2])
    assert isinstance(result_proba, tuple)
    for result, expected_array in zip(result_proba, expected_proba):
        assert np.allclose(result, expected_array)


def test_dummy_classifier_rejects_unsupported_or_invalid_inputs() -> None:
    from sciona.atoms.ml.sklearn.dummy import dummy_classifier_fit

    y = np.array([0.0, 1.0, 1.0], dtype=np.float64)
    with pytest.raises(Exception):
        dummy_classifier_fit(y, strategy="stratified")
    with pytest.raises(Exception):
        dummy_classifier_fit(y, strategy="uniform")
    with pytest.raises(Exception):
        dummy_classifier_fit(y, strategy="constant")
    with pytest.raises(Exception):
        dummy_classifier_fit(y, strategy="constant", constant=2.0)
