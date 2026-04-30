from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.multiclass import OneVsRestClassifier


def test_one_vs_rest_partial_fit_bookkeeping_imports() -> None:
    from sciona.atoms.ml.sklearn.multiclass.one_vs_rest_partial_fit_bookkeeping import (
        one_vs_rest_partial_fit_estimator_count,
        one_vs_rest_partial_fit_label_binarizer_classes,
        one_vs_rest_partial_fit_n_features_in,
    )

    assert callable(one_vs_rest_partial_fit_estimator_count)
    assert callable(one_vs_rest_partial_fit_label_binarizer_classes)
    assert callable(one_vs_rest_partial_fit_n_features_in)


def test_one_vs_rest_partial_fit_bookkeeping_matches_fitted_classifier_state() -> None:
    from sciona.atoms.ml.sklearn.multiclass.one_vs_rest_partial_fit_bookkeeping import (
        one_vs_rest_partial_fit_estimator_count,
        one_vs_rest_partial_fit_label_binarizer_classes,
        one_vs_rest_partial_fit_n_features_in,
    )

    X, y = load_iris(return_X_y=True)
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    clf = OneVsRestClassifier(
        SGDClassifier(loss="log_loss", max_iter=5, tol=None, random_state=0)
    )
    clf.partial_fit(X, y, classes=np.array([0, 1, 2], dtype=np.int64))

    assert one_vs_rest_partial_fit_estimator_count(clf.n_classes_) == len(clf.estimators_)
    assert np.array_equal(
        one_vs_rest_partial_fit_label_binarizer_classes(
            np.asarray(clf.classes_, dtype=np.float64)
        ),
        np.asarray(clf.label_binarizer_.classes_, dtype=np.float64),
    )
    assert one_vs_rest_partial_fit_n_features_in(clf.estimators_[0].n_features_in_) == clf.n_features_in_


def test_one_vs_rest_partial_fit_bookkeeping_rejects_invalid_inputs() -> None:
    from sciona.atoms.ml.sklearn.multiclass.one_vs_rest_partial_fit_bookkeeping import (
        one_vs_rest_partial_fit_estimator_count,
        one_vs_rest_partial_fit_label_binarizer_classes,
        one_vs_rest_partial_fit_n_features_in,
    )

    with pytest.raises((ViolationError, ValueError)):
        one_vs_rest_partial_fit_estimator_count(0)

    with pytest.raises((ViolationError, ValueError)):
        one_vs_rest_partial_fit_label_binarizer_classes(np.array([0.0, 0.0], dtype=np.float64))

    with pytest.raises((ViolationError, ValueError)):
        one_vs_rest_partial_fit_n_features_in(0)
