"""One-vs-rest postprocessing helpers adapted from scikit-learn."""

from __future__ import annotations

import array

import icontract
import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_one_vs_rest_binary_predict_threshold,
    witness_one_vs_rest_binary_probability_matrix,
    witness_one_vs_rest_decision_output,
    witness_one_vs_rest_multilabel_indicator_csc,
    witness_one_vs_rest_normalized_probability_matrix,
    witness_one_vs_rest_positive_probability_matrix,
)


def _finite_2d(values: NDArray[np.float64]) -> bool:
    array = np.asarray(values, dtype=np.float64)
    return bool(array.ndim == 2 and array.shape[0] >= 1 and array.shape[1] >= 1 and np.all(np.isfinite(array)))


def _finite_bool(value: bool) -> bool:
    return isinstance(value, bool)


def _positive_probability_stack_valid(values: NDArray[np.float64]) -> bool:
    array = np.asarray(values, dtype=np.float64)
    return bool(_finite_2d(values) and np.all((0.0 <= array) & (array <= 1.0)))


def _binary_probability_input_valid(values: NDArray[np.float64]) -> bool:
    array = np.asarray(values, dtype=np.float64)
    return bool(_positive_probability_stack_valid(values) and array.shape[1] == 1)


def _probability_rows_nonzero(values: NDArray[np.float64]) -> bool:
    array = np.asarray(values, dtype=np.float64)
    return bool(_positive_probability_stack_valid(values) and np.all(np.sum(array, axis=1) > 0.0))


def _decision_output_stack_valid(values: NDArray[np.float64]) -> bool:
    return _finite_2d(values)


def _threshold_valid(result: float) -> bool:
    return bool(isinstance(result, float) and np.isfinite(result) and result in {0.0, 0.5})


def _positive_probability_matrix_valid(result: NDArray[np.float64], values: NDArray[np.float64]) -> bool:
    output = np.asarray(result, dtype=np.float64)
    input_values = np.asarray(values, dtype=np.float64)
    return bool(output.shape == (input_values.shape[1], input_values.shape[0]) and np.all((0.0 <= output) & (output <= 1.0)))


def _binary_probability_matrix_valid(result: NDArray[np.float64], values: NDArray[np.float64]) -> bool:
    output = np.asarray(result, dtype=np.float64)
    input_values = np.asarray(values, dtype=np.float64)
    return bool(
        output.shape == (input_values.shape[0], 2)
        and np.all((0.0 <= output) & (output <= 1.0))
        and np.allclose(np.sum(output, axis=1), 1.0)
        and np.allclose(output[:, 1], input_values[:, 0])
    )


def _normalized_probability_matrix_valid(result: NDArray[np.float64], values: NDArray[np.float64]) -> bool:
    output = np.asarray(result, dtype=np.float64)
    input_values = np.asarray(values, dtype=np.float64)
    return bool(
        output.shape == input_values.shape
        and np.all(np.isfinite(output))
        and np.allclose(np.sum(output, axis=1), 1.0)
    )


def _indicator_csc_valid(result: sp.csc_matrix, responses: NDArray[np.float64]) -> bool:
    response_values = np.asarray(responses, dtype=np.float64)
    return bool(
        sp.isspmatrix_csc(result)
        and result.shape == response_values.shape
        and np.array_equal(result.data, np.ones(result.nnz, dtype=int))
    )


def _decision_output_valid(result: NDArray[np.float64], values: NDArray[np.float64]) -> bool:
    output = np.asarray(result, dtype=np.float64)
    input_values = np.asarray(values, dtype=np.float64)
    if input_values.shape[0] == 1:
        return bool(output.ndim == 1 and output.shape == (input_values.shape[1],) and np.all(np.isfinite(output)))
    return bool(output.shape == (input_values.shape[1], input_values.shape[0]) and np.all(np.isfinite(output)))


@register_atom(witness_one_vs_rest_binary_predict_threshold)
@icontract.require(lambda estimator_has_decision_function: _finite_bool(estimator_has_decision_function), "estimator_has_decision_function must be a bool")
@icontract.require(lambda estimator_is_classifier: _finite_bool(estimator_is_classifier), "estimator_is_classifier must be a bool")
@icontract.ensure(lambda result: _threshold_valid(result), "threshold must be sklearn's binary predict threshold")
def one_vs_rest_binary_predict_threshold(
    *,
    estimator_has_decision_function: bool,
    estimator_is_classifier: bool = True,
) -> float:
    """Resolve sklearn's binary OvR prediction threshold from estimator capability flags."""
    if estimator_has_decision_function and estimator_is_classifier:
        return 0.0
    return 0.5


@register_atom(witness_one_vs_rest_multilabel_indicator_csc)
@icontract.require(lambda responses: _finite_2d(responses), "responses must be a finite sample-by-class matrix")
@icontract.require(lambda threshold: isinstance(threshold, (int, float)) and not isinstance(threshold, bool) and np.isfinite(float(threshold)), "threshold must be finite")
@icontract.ensure(lambda result, responses: _indicator_csc_valid(result, responses), "indicator must be a CSC matrix aligned to the response matrix")
def one_vs_rest_multilabel_indicator_csc(
    responses: NDArray[np.float64],
    *,
    threshold: float = 0.0,
) -> sp.csc_matrix:
    """Build sklearn's multilabel CSC indicator matrix from OvR decision scores."""
    response_values = np.asarray(responses, dtype=np.float64)
    n_samples, n_classes = response_values.shape
    indices = array.array("i")
    indptr = array.array("i", [0])
    for class_index in range(n_classes):
        indices.extend(np.where(response_values[:, class_index] > float(threshold))[0])
        indptr.append(len(indices))
    data = np.ones(len(indices), dtype=int)
    return sp.csc_matrix((data, indices, indptr), shape=(n_samples, n_classes))


@register_atom(witness_one_vs_rest_positive_probability_matrix)
@icontract.require(lambda positive_class_probabilities: _positive_probability_stack_valid(positive_class_probabilities), "positive_class_probabilities must be a finite output-by-sample probability matrix")
@icontract.ensure(
    lambda result, positive_class_probabilities: _positive_probability_matrix_valid(result, positive_class_probabilities),
    "positive probability matrix must be sample by class",
)
def one_vs_rest_positive_probability_matrix(
    positive_class_probabilities: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Transpose per-estimator positive-class probabilities into sklearn's sample-by-class matrix."""
    return np.asarray(np.asarray(positive_class_probabilities, dtype=np.float64).T, dtype=np.float64)


@register_atom(witness_one_vs_rest_binary_probability_matrix)
@icontract.require(lambda probabilities: _binary_probability_input_valid(probabilities), "probabilities must be a finite sample-by-one matrix of positive-class probabilities")
@icontract.ensure(lambda result, probabilities: _binary_probability_matrix_valid(result, probabilities), "binary probability matrix must contain complementary negative and positive class columns")
def one_vs_rest_binary_probability_matrix(
    probabilities: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Expand sklearn's one-estimator OvR probability output into two binary class columns."""
    values = np.asarray(probabilities, dtype=np.float64)
    return np.asarray(np.concatenate((1.0 - values, values), axis=1), dtype=np.float64)


@register_atom(witness_one_vs_rest_normalized_probability_matrix)
@icontract.require(lambda probabilities: _probability_rows_nonzero(probabilities), "probabilities must be finite, nonnegative, and have strictly positive row sums")
@icontract.ensure(
    lambda result, probabilities: _normalized_probability_matrix_valid(result, probabilities),
    "normalized probabilities must preserve shape and sum to one row-wise",
)
def one_vs_rest_normalized_probability_matrix(
    probabilities: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Normalize OvR class probabilities row-wise for single-label multiclass output."""
    values = np.asarray(probabilities, dtype=np.float64)
    return np.asarray(values / np.sum(values, axis=1)[:, np.newaxis], dtype=np.float64)


@register_atom(witness_one_vs_rest_decision_output)
@icontract.require(lambda decision_outputs: _decision_output_stack_valid(decision_outputs), "decision_outputs must be a finite output-by-sample matrix")
@icontract.ensure(lambda result, decision_outputs: _decision_output_valid(result, decision_outputs), "decision output must follow sklearn's binary-vs-multiclass shape rule")
def one_vs_rest_decision_output(
    decision_outputs: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Shape OvR decision outputs as a vector for one estimator or a matrix otherwise."""
    values = np.asarray(decision_outputs, dtype=np.float64)
    if values.shape[0] == 1:
        return np.asarray(values[0], dtype=np.float64)
    return np.asarray(values.T, dtype=np.float64)
