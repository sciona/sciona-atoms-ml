from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError
from sklearn.multiclass import OutputCodeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC

from sciona.atoms.ml.sklearn.multiclass.output_code_book import (
    output_code_discrete_book,
    output_code_uniform_book,
)


def test_output_code_book_atoms_import() -> None:
    assert callable(output_code_uniform_book)
    assert callable(output_code_discrete_book)


def test_output_code_uniform_book_is_seeded_and_shaped() -> None:
    observed = output_code_uniform_book(n_classes=3, n_estimators=5, seed=7)
    expected = np.random.RandomState(7).uniform(size=(3, 5))
    assert observed.shape == (3, 5)
    assert np.allclose(observed, expected)
    assert np.all((observed >= 0.0) & (observed < 1.0))


def test_output_code_discrete_book_matches_sklearn_threshold_rules() -> None:
    uniform = np.array([[0.1, 0.9, 0.5], [0.6, 0.2, 0.8]], dtype=np.float64)

    decision_book = output_code_discrete_book(uniform, has_decision_function=True)
    probability_book = output_code_discrete_book(uniform, has_decision_function=False)

    assert np.array_equal(
        decision_book,
        np.array([[-1.0, 1.0, -1.0], [1.0, -1.0, 1.0]], dtype=np.float64),
    )
    assert np.array_equal(
        probability_book,
        np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]], dtype=np.float64),
    )


def test_output_code_book_atoms_match_sklearn_code_book_for_decision_estimators() -> None:
    X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]], dtype=np.float64)
    y = np.array([0, 1, 2, 0, 1, 2], dtype=np.int64)
    clf = OutputCodeClassifier(LinearSVC(random_state=0), code_size=1.5, random_state=11)
    clf.fit(X, y)

    uniform = output_code_uniform_book(n_classes=3, n_estimators=4, seed=11)
    observed = output_code_discrete_book(uniform, has_decision_function=True)

    assert np.array_equal(observed, clf.code_book_)


def test_output_code_book_atoms_match_sklearn_code_book_for_probability_estimators() -> None:
    X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]], dtype=np.float64)
    y = np.array([0, 1, 2, 0, 1, 2], dtype=np.int64)
    clf = OutputCodeClassifier(GaussianNB(), code_size=1.5, random_state=11)
    clf.fit(X, y)

    uniform = output_code_uniform_book(n_classes=3, n_estimators=4, seed=11)
    observed = output_code_discrete_book(uniform, has_decision_function=False)

    assert np.array_equal(observed, clf.code_book_)


def test_output_code_book_atoms_reject_invalid_inputs() -> None:
    with pytest.raises(ViolationError):
        output_code_uniform_book(n_classes=0, n_estimators=2, seed=1)

    with pytest.raises(ViolationError):
        output_code_uniform_book(n_classes=2, n_estimators=0, seed=1)

    with pytest.raises(ViolationError):
        output_code_discrete_book(np.array([[1.2, 0.1]], dtype=np.float64), has_decision_function=True)
