from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.multiclass import _threshold_for_binary_predict


class DecisionClassifier(ClassifierMixin, BaseEstimator):
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        del X
        return np.array([0.0], dtype=np.float64)


class ProbabilityOnlyClassifier(ClassifierMixin, BaseEstimator):
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        del X
        return np.array([[0.25, 0.75]], dtype=np.float64)


class DecisionRegressor(RegressorMixin, BaseEstimator):
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        del X
        return np.array([0.0], dtype=np.float64)


def test_one_vs_rest_postprocessing_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.multiclass.one_vs_rest_postprocessing import (
        one_vs_rest_binary_predict_threshold,
        one_vs_rest_binary_probability_matrix,
        one_vs_rest_decision_output,
        one_vs_rest_multilabel_indicator_csc,
        one_vs_rest_normalized_probability_matrix,
        one_vs_rest_positive_probability_matrix,
    )

    assert callable(one_vs_rest_binary_predict_threshold)
    assert callable(one_vs_rest_multilabel_indicator_csc)
    assert callable(one_vs_rest_positive_probability_matrix)
    assert callable(one_vs_rest_binary_probability_matrix)
    assert callable(one_vs_rest_normalized_probability_matrix)
    assert callable(one_vs_rest_decision_output)


def test_binary_predict_threshold_matches_sklearn_helper() -> None:
    from sciona.atoms.ml.sklearn.multiclass.one_vs_rest_postprocessing import one_vs_rest_binary_predict_threshold

    assert one_vs_rest_binary_predict_threshold(estimator_has_decision_function=True, estimator_is_classifier=True) == _threshold_for_binary_predict(DecisionClassifier())
    assert one_vs_rest_binary_predict_threshold(estimator_has_decision_function=False, estimator_is_classifier=True) == _threshold_for_binary_predict(ProbabilityOnlyClassifier())
    assert one_vs_rest_binary_predict_threshold(estimator_has_decision_function=True, estimator_is_classifier=False) == _threshold_for_binary_predict(DecisionRegressor())


def test_multilabel_indicator_csc_matches_thresholded_scores() -> None:
    from sciona.atoms.ml.sklearn.multiclass.one_vs_rest_postprocessing import one_vs_rest_multilabel_indicator_csc

    responses = np.array(
        [
            [0.2, -0.1, 0.7],
            [0.6, 0.8, -0.5],
            [-0.2, 0.4, 0.3],
        ],
        dtype=np.float64,
    )

    result = one_vs_rest_multilabel_indicator_csc(responses, threshold=0.25)

    expected = sp.csc_matrix((responses > 0.25).astype(int))
    assert sp.isspmatrix_csc(result)
    assert np.array_equal(result.toarray(), expected.toarray())


def test_positive_probability_matrix_transposes_output_by_sample_stack() -> None:
    from sciona.atoms.ml.sklearn.multiclass.one_vs_rest_postprocessing import one_vs_rest_positive_probability_matrix

    positive_probs = np.array(
        [
            [0.1, 0.3, 0.9],
            [0.6, 0.2, 0.4],
        ],
        dtype=np.float64,
    )

    result = one_vs_rest_positive_probability_matrix(positive_probs)

    assert np.array_equal(result, positive_probs.T)


def test_binary_probability_matrix_adds_complementary_column() -> None:
    from sciona.atoms.ml.sklearn.multiclass.one_vs_rest_postprocessing import one_vs_rest_binary_probability_matrix

    probabilities = np.array([[0.8], [0.25], [0.5]], dtype=np.float64)

    result = one_vs_rest_binary_probability_matrix(probabilities)

    expected = np.array([[0.2, 0.8], [0.75, 0.25], [0.5, 0.5]], dtype=np.float64)
    assert np.allclose(result, expected)


def test_normalized_probability_matrix_normalizes_rows() -> None:
    from sciona.atoms.ml.sklearn.multiclass.one_vs_rest_postprocessing import one_vs_rest_normalized_probability_matrix

    probabilities = np.array([[0.2, 0.3, 0.1], [0.1, 0.1, 0.2]], dtype=np.float64)

    result = one_vs_rest_normalized_probability_matrix(probabilities)

    assert np.allclose(result, np.array([[1.0 / 3.0, 0.5, 1.0 / 6.0], [0.25, 0.25, 0.5]], dtype=np.float64))


def test_decision_output_uses_binary_vector_or_multiclass_matrix_shape() -> None:
    from sciona.atoms.ml.sklearn.multiclass.one_vs_rest_postprocessing import one_vs_rest_decision_output

    binary_outputs = np.array([[0.1, -0.2, 0.4]], dtype=np.float64)
    multiclass_outputs = np.array([[0.1, -0.2], [0.3, 0.4], [0.0, 0.9]], dtype=np.float64)

    binary_result = one_vs_rest_decision_output(binary_outputs)
    multiclass_result = one_vs_rest_decision_output(multiclass_outputs)

    assert np.array_equal(binary_result, np.array([0.1, -0.2, 0.4], dtype=np.float64))
    assert np.array_equal(multiclass_result, multiclass_outputs.T)
