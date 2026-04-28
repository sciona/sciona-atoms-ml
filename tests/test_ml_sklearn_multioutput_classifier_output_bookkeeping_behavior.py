from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor

from sciona.atoms.ml.sklearn.multioutput.classifier_output_bookkeeping import (
    multioutput_classifier_probability_blocks,
    multioutput_classifier_score_require_2d_targets,
    multioutput_classifier_score_require_matching_output_count,
    multioutput_predict_require_base_predict_method,
)


class NoPredictEstimator(BaseEstimator):
    pass


def test_multioutput_classifier_output_bookkeeping_atoms_import() -> None:
    assert callable(multioutput_predict_require_base_predict_method)
    assert callable(multioutput_classifier_score_require_2d_targets)
    assert callable(multioutput_classifier_score_require_matching_output_count)
    assert callable(multioutput_classifier_probability_blocks)


def test_predict_guard_accepts_and_rejects_predict_support() -> None:
    assert multioutput_predict_require_base_predict_method(True) is True
    with pytest.raises(ValueError, match="base estimator should implement a predict method"):
        multioutput_predict_require_base_predict_method(False)


def test_score_2d_target_guard_matches_multioutput_classifier() -> None:
    y = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    clf = MultiOutputClassifier(DummyClassifier())
    clf.estimators_ = [DummyClassifier(), DummyClassifier()]

    with pytest.raises(ValueError, match="multi target classification but has only one"):
        multioutput_classifier_score_require_2d_targets(y)

    with pytest.raises(ValueError, match="multi target classification but has only one"):
        clf.score(np.zeros((3, 1)), y)


def test_score_matching_output_count_guard_matches_multioutput_classifier() -> None:
    y = np.array([[0.0], [1.0], [0.0]], dtype=np.float64)
    clf = MultiOutputClassifier(DummyClassifier())
    clf.estimators_ = [DummyClassifier(), DummyClassifier()]

    with pytest.raises(ValueError, match="The number of outputs of Y for fit 2 and score 1 should be same"):
        multioutput_classifier_score_require_matching_output_count(y, 2)

    with pytest.raises(ValueError, match="The number of outputs of Y for fit 2 and score 1 should be same"):
        clf.score(np.zeros((3, 1)), y)


def test_probability_blocks_match_multioutput_classifier_predict_proba_structure() -> None:
    X = np.array([[0.0], [1.0], [0.0], [1.0]], dtype=np.float64)
    y = np.array([[0, 1], [1, 0], [0, 1], [1, 0]], dtype=np.int64)
    clf = MultiOutputClassifier(DummyClassifier(strategy="prior")).fit(X, y)

    observed = multioutput_classifier_probability_blocks(tuple(clf.predict_proba(X)))
    expected = tuple(clf.predict_proba(X))

    assert isinstance(observed, tuple)
    assert len(observed) == len(expected) == 2
    for obs_block, exp_block in zip(observed, expected):
        assert np.array_equal(obs_block, exp_block)


def test_contract_rejects_invalid_probability_blocks() -> None:
    with pytest.raises(ViolationError):
        multioutput_classifier_probability_blocks((np.array([0.2, 0.8], dtype=np.float64),))


def test_multioutput_regressor_predict_raises_without_predict_method() -> None:
    reg = MultiOutputRegressor(NoPredictEstimator())
    reg.estimators_ = [NoPredictEstimator()]

    with pytest.raises(ValueError, match="base estimator should implement a predict method"):
        reg.predict(np.zeros((2, 1), dtype=np.float64))
