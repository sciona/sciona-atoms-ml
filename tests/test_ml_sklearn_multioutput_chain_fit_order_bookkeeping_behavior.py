from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import ClassifierChain


def test_multioutput_chain_fit_order_bookkeeping_imports() -> None:
    from sciona.atoms.ml.sklearn.multioutput.chain_fit_order_bookkeeping import (
        chain_fit_log_message,
        chain_fit_require_valid_order,
        chain_fit_tuple_order_array,
    )

    assert callable(chain_fit_tuple_order_array)
    assert callable(chain_fit_require_valid_order)
    assert callable(chain_fit_log_message)


def test_chain_fit_tuple_order_array_matches_classifier_chain_tuple_coercion() -> None:
    from sciona.atoms.ml.sklearn.multioutput.chain_fit_order_bookkeeping import chain_fit_tuple_order_array

    X = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float64)
    y = np.array([[0, 1], [1, 0], [0, 1], [1, 0]], dtype=np.int64)
    clf = ClassifierChain(LogisticRegression(max_iter=1000), order=(1, 0), random_state=0)
    clf.fit(X, y)

    result = chain_fit_tuple_order_array((1, 0))
    assert isinstance(clf.order_, np.ndarray)
    assert np.array_equal(result, clf.order_)


def test_chain_fit_require_valid_order_matches_classifier_chain_validation() -> None:
    from sciona.atoms.ml.sklearn.multioutput.chain_fit_order_bookkeeping import chain_fit_require_valid_order

    assert chain_fit_require_valid_order([1, 0], 2) is True
    assert chain_fit_require_valid_order(np.array([2, 0, 1], dtype=np.int64), 3) is True

    X = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float64)
    y = np.array([[0, 1], [1, 0], [0, 1], [1, 0]], dtype=np.int64)
    clf = ClassifierChain(LogisticRegression(max_iter=1000), order=[0, 0], random_state=0)
    with pytest.raises(ValueError, match="invalid order"):
        clf.fit(X, y)

    with pytest.raises(ValueError, match="invalid order"):
        chain_fit_require_valid_order([0, 0], 2)


def test_chain_fit_log_message_matches_private_helper_formatting() -> None:
    from sciona.atoms.ml.sklearn.multioutput.chain_fit_order_bookkeeping import chain_fit_log_message

    clf = ClassifierChain(LogisticRegression(max_iter=1000), verbose=True)
    assert chain_fit_log_message(
        True,
        estimator_idx=2,
        n_estimators=4,
        processing_msg="Processing order 3",
    ) == clf._log_message(estimator_idx=2, n_estimators=4, processing_msg="Processing order 3")

    quiet = ClassifierChain(LogisticRegression(max_iter=1000), verbose=False)
    assert chain_fit_log_message(
        False,
        estimator_idx=1,
        n_estimators=2,
        processing_msg="Processing order 0",
    ) is quiet._log_message(estimator_idx=1, n_estimators=2, processing_msg="Processing order 0")


def test_multioutput_chain_fit_order_bookkeeping_contracts_reject_invalid_inputs() -> None:
    from sciona.atoms.ml.sklearn.multioutput.chain_fit_order_bookkeeping import (
        chain_fit_log_message,
        chain_fit_require_valid_order,
        chain_fit_tuple_order_array,
    )

    with pytest.raises(ViolationError):
        chain_fit_tuple_order_array((0, True))

    with pytest.raises(ViolationError):
        chain_fit_require_valid_order([0, 1], 0)

    with pytest.raises(ViolationError):
        chain_fit_log_message(True, estimator_idx=3, n_estimators=2, processing_msg="Processing order 1")
