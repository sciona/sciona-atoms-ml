from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError
from sklearn.linear_model import SGDClassifier
from sklearn.multioutput import MultiOutputClassifier


def test_multioutput_partial_fit_bookkeeping_imports() -> None:
    from sciona.atoms.ml.sklearn.multioutput.partial_fit_bookkeeping import (
        multioutput_partial_fit_class_vector,
        multioutput_partial_fit_first_call,
        multioutput_partial_fit_use_base_estimator,
    )

    assert callable(multioutput_partial_fit_first_call)
    assert callable(multioutput_partial_fit_use_base_estimator)
    assert callable(multioutput_partial_fit_class_vector)


def test_multioutput_partial_fit_first_call_and_base_estimator_flag_match_state() -> None:
    from sciona.atoms.ml.sklearn.multioutput.partial_fit_bookkeeping import (
        multioutput_partial_fit_first_call,
        multioutput_partial_fit_use_base_estimator,
    )

    clf = MultiOutputClassifier(SGDClassifier(loss="hinge", random_state=0, max_iter=1, tol=None))
    first_time = multioutput_partial_fit_first_call(has_estimators=hasattr(clf, "estimators_"))
    assert first_time is True
    assert multioutput_partial_fit_use_base_estimator(first_time) is True

    X = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float64)
    y = np.array([[0, 1], [1, 0], [0, 1], [1, 0]], dtype=np.int64)
    classes = [np.array([0, 1], dtype=np.int64), np.array([0, 1], dtype=np.int64)]
    clf.partial_fit(X, y, classes=classes)

    first_time = multioutput_partial_fit_first_call(has_estimators=hasattr(clf, "estimators_"))
    assert first_time is False
    assert multioutput_partial_fit_use_base_estimator(first_time) is False


def test_multioutput_partial_fit_class_vector_matches_per_output_routing() -> None:
    from sciona.atoms.ml.sklearn.multioutput.partial_fit_bookkeeping import multioutput_partial_fit_class_vector

    classes = (
        np.array([0.0, 1.0], dtype=np.float64),
        np.array([1.0, 2.0, 3.0], dtype=np.float64),
    )

    assert np.array_equal(
        multioutput_partial_fit_class_vector(classes, 0),
        classes[0],
    )
    assert np.array_equal(
        multioutput_partial_fit_class_vector(classes, 1),
        classes[1],
    )
    assert multioutput_partial_fit_class_vector(None, 0) is None


def test_multioutput_partial_fit_bookkeeping_contracts_reject_invalid_inputs() -> None:
    from sciona.atoms.ml.sklearn.multioutput.partial_fit_bookkeeping import (
        multioutput_partial_fit_class_vector,
        multioutput_partial_fit_first_call,
        multioutput_partial_fit_use_base_estimator,
    )

    with pytest.raises(ViolationError):
        multioutput_partial_fit_first_call(has_estimators=1)

    with pytest.raises(ViolationError):
        multioutput_partial_fit_use_base_estimator("yes")

    with pytest.raises(ViolationError):
        multioutput_partial_fit_class_vector((np.array([0.0, 1.0]),), 2)
