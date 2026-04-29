from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError
from sklearn.linear_model import SGDClassifier
from sklearn.multiclass import OneVsRestClassifier


def test_one_vs_rest_fit_bookkeeping_imports() -> None:
    from sciona.atoms.ml.sklearn.multiclass.one_vs_rest_fit_bookkeeping import (
        one_vs_rest_binary_fit_labels,
        one_vs_rest_class_count,
        one_vs_rest_multilabel_flag,
        one_vs_rest_partial_fit_first_call,
    )

    assert callable(one_vs_rest_binary_fit_labels)
    assert callable(one_vs_rest_class_count)
    assert callable(one_vs_rest_multilabel_flag)
    assert callable(one_vs_rest_partial_fit_first_call)


def test_one_vs_rest_binary_fit_labels_match_fit_classes_argument() -> None:
    from sciona.atoms.ml.sklearn.multiclass.one_vs_rest_fit_bookkeeping import one_vs_rest_binary_fit_labels

    observed = one_vs_rest_binary_fit_labels(3)

    assert np.array_equal(observed, np.asarray(["not 3", 3], dtype=object))


def test_one_vs_rest_class_count_and_multilabel_flag_match_fitted_classifier() -> None:
    from sciona.atoms.ml.sklearn.multiclass.one_vs_rest_fit_bookkeeping import (
        one_vs_rest_class_count,
        one_vs_rest_multilabel_flag,
    )

    X = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float64)
    y = np.array([0, 1, 2, 1], dtype=np.int64)
    clf = OneVsRestClassifier(SGDClassifier(loss="hinge", random_state=0, max_iter=5, tol=None))
    clf.fit(X, y)

    assert one_vs_rest_class_count(tuple(clf.classes_.tolist())) == clf.n_classes_
    assert one_vs_rest_multilabel_flag(clf.label_binarizer_.y_type_) is clf.multilabel_


def test_one_vs_rest_multilabel_flag_matches_multilabel_fit() -> None:
    from sciona.atoms.ml.sklearn.multiclass.one_vs_rest_fit_bookkeeping import one_vs_rest_multilabel_flag

    X = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float64)
    y = np.array([[1, 0], [0, 1], [1, 1], [0, 0]], dtype=np.int64)
    clf = OneVsRestClassifier(SGDClassifier(loss="hinge", random_state=0, max_iter=5, tol=None))
    clf.fit(X, y)

    assert one_vs_rest_multilabel_flag(clf.label_binarizer_.y_type_) is True
    assert clf.multilabel_ is True


def test_one_vs_rest_partial_fit_first_call_matches_estimator_state() -> None:
    from sciona.atoms.ml.sklearn.multiclass.one_vs_rest_fit_bookkeeping import one_vs_rest_partial_fit_first_call

    clf = OneVsRestClassifier(SGDClassifier(loss="hinge", random_state=0, max_iter=1, tol=None))
    assert one_vs_rest_partial_fit_first_call(has_estimators=hasattr(clf, "estimators_")) is True

    X = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float64)
    y = np.array([0, 1, 2, 1], dtype=np.int64)
    clf.partial_fit(X, y, classes=np.array([0, 1, 2], dtype=np.int64))
    assert one_vs_rest_partial_fit_first_call(has_estimators=hasattr(clf, "estimators_")) is False


def test_one_vs_rest_fit_bookkeeping_contracts_reject_invalid_inputs() -> None:
    from sciona.atoms.ml.sklearn.multiclass.one_vs_rest_fit_bookkeeping import (
        one_vs_rest_binary_fit_labels,
        one_vs_rest_class_count,
        one_vs_rest_multilabel_flag,
    )

    with pytest.raises(ViolationError):
        one_vs_rest_binary_fit_labels(np.inf)

    with pytest.raises(ViolationError):
        one_vs_rest_class_count(())

    with pytest.raises(ViolationError):
        one_vs_rest_multilabel_flag("")
