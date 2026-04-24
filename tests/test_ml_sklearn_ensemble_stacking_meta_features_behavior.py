from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
from icontract import ViolationError
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import StackingClassifier


def test_stacking_meta_features_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.ensemble.stacking_meta_features import (
        stacking_feature_names_out,
        stacking_meta_feature_matrix,
        stacking_meta_feature_widths,
    )

    assert callable(stacking_meta_feature_matrix)
    assert callable(stacking_meta_feature_widths)
    assert callable(stacking_feature_names_out)


def test_stacking_meta_feature_matrix_matches_sklearn_binary_probability_and_predict() -> None:
    from sciona.atoms.ml.sklearn.ensemble.stacking_meta_features import (
        stacking_meta_feature_matrix,
        stacking_meta_feature_widths,
    )

    predictions = (
        np.array([[0.2, 0.8], [0.6, 0.4], [0.1, 0.9]], dtype=np.float64),
        np.array([1.0, 0.0, 1.0], dtype=np.float64),
    )
    stack_method_names = ("predict_proba", "predict")

    estimator = StackingClassifier(
        estimators=[("a", DummyClassifier()), ("b", DummyClassifier())]
    )
    estimator.stack_method_ = list(stack_method_names)
    estimator.classes_ = np.array([0, 1], dtype=np.int64)
    estimator.passthrough = False

    expected = estimator._concatenate_predictions(np.zeros((3, 1), dtype=np.float64), list(predictions))
    expected_widths = np.asarray(estimator._n_feature_outs, dtype=np.int64)

    result = stacking_meta_feature_matrix(
        predictions,
        stack_method_names,
        is_binary_classification=True,
    )
    widths = stacking_meta_feature_widths(
        predictions,
        stack_method_names,
        is_binary_classification=True,
    )

    assert np.allclose(result, expected)
    assert np.array_equal(widths, expected_widths)


def test_stacking_meta_feature_matrix_matches_sklearn_multilabel_probability_lists() -> None:
    from sciona.atoms.ml.sklearn.ensemble.stacking_meta_features import (
        stacking_meta_feature_matrix,
        stacking_meta_feature_widths,
    )

    predictions = (
        (
            np.array([[0.2, 0.8], [0.6, 0.4]], dtype=np.float64),
            np.array([[0.1, 0.9], [0.7, 0.3]], dtype=np.float64),
        ),
    )
    stack_method_names = ("predict_proba",)

    estimator = StackingClassifier(estimators=[("a", DummyClassifier())])
    estimator.stack_method_ = list(stack_method_names)
    estimator.classes_ = [np.array([0, 1], dtype=np.int64), np.array([0, 1], dtype=np.int64)]
    estimator.passthrough = False

    expected = estimator._concatenate_predictions(np.zeros((2, 1), dtype=np.float64), [list(predictions[0])])
    expected_widths = np.asarray(estimator._n_feature_outs, dtype=np.int64)

    result = stacking_meta_feature_matrix(
        predictions,
        stack_method_names,
        is_binary_classification=False,
    )
    widths = stacking_meta_feature_widths(
        predictions,
        stack_method_names,
        is_binary_classification=False,
    )

    assert np.allclose(result, expected)
    assert np.array_equal(widths, expected_widths)


def test_stacking_meta_feature_matrix_matches_sklearn_sparse_passthrough() -> None:
    from sciona.atoms.ml.sklearn.ensemble.stacking_meta_features import (
        stacking_meta_feature_matrix,
        stacking_meta_feature_widths,
    )

    X = sp.csr_matrix([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    predictions = (
        np.array([[0.3, 0.7], [0.4, 0.6]], dtype=np.float64),
    )
    stack_method_names = ("predict_proba",)

    estimator = StackingClassifier(estimators=[("a", DummyClassifier())], passthrough=True)
    estimator.stack_method_ = list(stack_method_names)
    estimator.classes_ = np.array([0, 1], dtype=np.int64)
    estimator.passthrough = True

    expected = estimator._concatenate_predictions(X, list(predictions))
    expected_widths = np.asarray(estimator._n_feature_outs, dtype=np.int64)

    result = stacking_meta_feature_matrix(
        predictions,
        stack_method_names,
        is_binary_classification=True,
        X=X,
        passthrough=True,
    )
    widths = stacking_meta_feature_widths(
        predictions,
        stack_method_names,
        is_binary_classification=True,
    )

    assert sp.issparse(result)
    assert result.getformat() == X.getformat()
    assert np.allclose(result.toarray(), expected.toarray())
    assert np.array_equal(widths, expected_widths)


def test_stacking_feature_names_out_matches_sklearn() -> None:
    from sciona.atoms.ml.sklearn.ensemble.stacking_meta_features import stacking_feature_names_out

    class _Fitted:
        n_features_in_ = 2

    estimator = StackingClassifier(
        estimators=[("a", DummyClassifier()), ("b", DummyClassifier())],
        passthrough=True,
    )
    estimator.estimators_ = [_Fitted(), _Fitted()]
    estimator.estimators = [("a", DummyClassifier()), ("b", DummyClassifier())]
    estimator._n_feature_outs = [1, 2]
    estimator.passthrough = True
    estimator.feature_names_in_ = np.array(["x0", "x1"], dtype=object)

    expected = estimator.get_feature_names_out()
    result = stacking_feature_names_out(
        "stackingclassifier",
        ("a", "b"),
        np.array([1, 2], dtype=np.int64),
        input_features=("x0", "x1"),
        passthrough=True,
    )

    assert np.array_equal(result, expected)


def test_contracts_reject_invalid_stacking_meta_feature_inputs() -> None:
    from sciona.atoms.ml.sklearn.ensemble.stacking_meta_features import (
        stacking_feature_names_out,
        stacking_meta_feature_matrix,
        stacking_meta_feature_widths,
    )

    with pytest.raises(ViolationError):
        stacking_meta_feature_matrix(
            (np.array([[1.0]], dtype=np.float64),),
            ("predict_proba",),
            is_binary_classification=True,
        )

    with pytest.raises(ViolationError):
        stacking_meta_feature_widths(
            (np.array([1.0, 2.0], dtype=np.float64),),
            ("unknown",),
            is_binary_classification=False,
        )

    with pytest.raises(ViolationError):
        stacking_meta_feature_matrix(
            (np.array([1.0, 2.0], dtype=np.float64),),
            ("predict",),
            is_binary_classification=False,
            X=np.array([[1.0], [2.0], [3.0]], dtype=np.float64),
            passthrough=True,
        )

    with pytest.raises(ViolationError):
        stacking_feature_names_out(
            "stackingclassifier",
            ("a", "b"),
            np.array([1], dtype=np.int64),
        )
