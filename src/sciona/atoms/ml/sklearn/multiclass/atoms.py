"""Estimator-independent multiclass helper atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_one_vs_one_class_pairs,
    witness_one_vs_one_decision_scores,
    witness_one_vs_rest_binary_indicator,
    witness_one_vs_rest_multiclass_labels,
    witness_output_code_book,
    witness_output_code_decode,
)


def _classes_valid(classes: NDArray[np.float64], *, min_classes: int = 1) -> bool:
    values = np.asarray(classes, dtype=np.float64)
    return bool(
        values.ndim == 1
        and values.shape[0] >= min_classes
        and np.all(np.isfinite(values))
        and np.unique(values).shape[0] == values.shape[0]
    )


def _responses_match_classes(responses: NDArray[np.float64], classes: NDArray[np.float64]) -> bool:
    values = np.asarray(responses, dtype=np.float64)
    class_values = np.asarray(classes, dtype=np.float64)
    return bool(
        values.ndim == 2
        and values.shape[0] >= 1
        and values.shape[1] >= 1
        and np.all(np.isfinite(values))
        and _classes_valid(classes)
        and values.shape[1] == class_values.shape[0]
    )


def _indicator_responses_valid(responses: NDArray[np.float64]) -> bool:
    values = np.asarray(responses, dtype=np.float64)
    return bool(values.ndim == 2 and values.shape[0] >= 1 and values.shape[1] >= 1 and np.all(np.isfinite(values)))


def _finite_float(value: float) -> bool:
    return bool(isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(float(value)))


def _binary_prediction_matrix_valid(predictions: NDArray[np.int64], n_classes: int) -> bool:
    values = np.asarray(predictions)
    return bool(
        isinstance(n_classes, int)
        and not isinstance(n_classes, bool)
        and n_classes >= 2
        and values.ndim == 2
        and values.shape[0] >= 1
        and values.shape[1] == n_classes * (n_classes - 1) // 2
        and np.issubdtype(values.dtype, np.integer)
        and np.all((values == 0) | (values == 1))
    )


def _confidence_matrix_valid(confidences: NDArray[np.float64], predictions: NDArray[np.int64]) -> bool:
    values = np.asarray(confidences, dtype=np.float64)
    prediction_values = np.asarray(predictions)
    return bool(values.ndim == 2 and values.shape == prediction_values.shape and np.all(np.isfinite(values)))


def _class_pair_count_valid(n_classes: int) -> bool:
    return bool(isinstance(n_classes, int) and not isinstance(n_classes, bool) and n_classes >= 2)


def _code_book_params_valid(n_classes: int, code_size: float, random_state: int) -> bool:
    return bool(
        isinstance(n_classes, int)
        and not isinstance(n_classes, bool)
        and n_classes >= 1
        and _finite_float(code_size)
        and float(code_size) > 0.0
        and int(n_classes * float(code_size)) >= 1
        and isinstance(random_state, int)
        and not isinstance(random_state, bool)
    )


def _code_book_valid(result: NDArray[np.float64], n_classes: int, code_size: float, estimator_has_decision_function: bool) -> bool:
    values = np.asarray(result, dtype=np.float64)
    low_value = -1.0 if estimator_has_decision_function else 0.0
    return bool(
        values.shape == (n_classes, int(n_classes * float(code_size)))
        and np.all(np.isin(values, np.array([low_value, 1.0], dtype=np.float64)))
    )


def _decode_inputs_valid(responses: NDArray[np.float64], code_book: NDArray[np.float64], classes: NDArray[np.float64]) -> bool:
    response_values = np.asarray(responses, dtype=np.float64)
    code_values = np.asarray(code_book, dtype=np.float64)
    class_values = np.asarray(classes, dtype=np.float64)
    return bool(
        response_values.ndim == 2
        and code_values.ndim == 2
        and response_values.shape[0] >= 1
        and response_values.shape[1] >= 1
        and response_values.shape[1] == code_values.shape[1]
        and code_values.shape[0] >= 1
        and class_values.shape[0] == code_values.shape[0]
        and np.all(np.isfinite(response_values))
        and np.all(np.isfinite(code_values))
        and _classes_valid(classes)
    )


def _labels_valid(result: NDArray[np.float64], n_samples: int, classes: NDArray[np.float64]) -> bool:
    values = np.asarray(result, dtype=np.float64)
    class_values = np.asarray(classes, dtype=np.float64)
    return bool(values.shape == (n_samples,) and np.all(np.isin(values, class_values)))


def _indicator_valid(result: NDArray[np.bool_], responses: NDArray[np.float64]) -> bool:
    values = np.asarray(result)
    response_values = np.asarray(responses)
    return bool(values.dtype == np.bool_ and values.shape == response_values.shape)


def _ovo_scores_valid(result: NDArray[np.float64], predictions: NDArray[np.int64], n_classes: int) -> bool:
    values = np.asarray(result, dtype=np.float64)
    prediction_values = np.asarray(predictions)
    return bool(values.shape == (prediction_values.shape[0], n_classes) and np.all(np.isfinite(values)))


def _pairs_valid(result: NDArray[np.int64], n_classes: int) -> bool:
    values = np.asarray(result)
    return bool(
        values.shape == (n_classes * (n_classes - 1) // 2, 2)
        and np.issubdtype(values.dtype, np.integer)
        and np.all(values[:, 0] < values[:, 1])
        and np.array_equal(values, np.asarray([(i, j) for i in range(n_classes) for j in range(i + 1, n_classes)], dtype=np.int64))
    )


@register_atom(witness_one_vs_rest_multiclass_labels)
@icontract.require(lambda responses, classes: _responses_match_classes(responses, classes), "responses must be a finite sample-by-class matrix")
@icontract.ensure(lambda result, responses, classes: _labels_valid(result, np.asarray(responses).shape[0], classes), "labels must come from the class vector")
def one_vs_rest_multiclass_labels(
    responses: NDArray[np.float64],
    classes: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Choose the class with the largest one-vs-rest binary response per sample."""
    response_values = np.asarray(responses, dtype=np.float64)
    class_values = np.asarray(classes, dtype=np.float64)
    return np.asarray(class_values[np.argmax(response_values, axis=1)], dtype=np.float64)


@register_atom(witness_one_vs_rest_binary_indicator)
@icontract.require(lambda responses: _indicator_responses_valid(responses), "responses must be a finite sample-by-class matrix")
@icontract.require(lambda threshold: _finite_float(threshold), "threshold must be finite")
@icontract.ensure(lambda result, responses: _indicator_valid(result, responses), "indicator must match the response matrix shape")
def one_vs_rest_binary_indicator(
    responses: NDArray[np.float64],
    *,
    threshold: float = 0.0,
) -> NDArray[np.bool_]:
    """Threshold one-vs-rest binary responses into a multilabel indicator matrix."""
    return np.asarray(np.asarray(responses, dtype=np.float64) > float(threshold), dtype=np.bool_)


@register_atom(witness_one_vs_one_decision_scores)
@icontract.require(lambda predictions, n_classes: _binary_prediction_matrix_valid(predictions, n_classes), "binary predictions must follow one-vs-one pair order")
@icontract.require(lambda confidences, predictions: _confidence_matrix_valid(confidences, predictions), "confidences must match binary predictions")
@icontract.ensure(lambda result, predictions, n_classes: _ovo_scores_valid(result, predictions, n_classes), "decision scores must have one column per class")
def one_vs_one_decision_scores(
    predictions: NDArray[np.int64],
    confidences: NDArray[np.float64],
    *,
    n_classes: int,
) -> NDArray[np.float64]:
    """Compute sklearn's OvO vote scores with confidence-based tie breaking."""
    prediction_values = np.asarray(predictions, dtype=np.int64)
    confidence_values = np.asarray(confidences, dtype=np.float64)
    n_samples = prediction_values.shape[0]
    votes = np.zeros((n_samples, n_classes), dtype=np.float64)
    confidence_sums = np.zeros((n_samples, n_classes), dtype=np.float64)

    k = 0
    for i in range(n_classes):
        for j in range(i + 1, n_classes):
            confidence_sums[:, i] -= confidence_values[:, k]
            confidence_sums[:, j] += confidence_values[:, k]
            votes[prediction_values[:, k] == 0, i] += 1.0
            votes[prediction_values[:, k] == 1, j] += 1.0
            k += 1

    transformed = confidence_sums / (3.0 * (np.abs(confidence_sums) + 1.0))
    return np.asarray(votes + transformed, dtype=np.float64)


@register_atom(witness_one_vs_one_class_pairs)
@icontract.require(lambda n_classes: _class_pair_count_valid(n_classes), "n_classes must be at least 2")
@icontract.ensure(lambda result, n_classes: _pairs_valid(result, n_classes), "class pairs must follow sklearn's nested-loop order")
def one_vs_one_class_pairs(n_classes: int) -> NDArray[np.int64]:
    """Return the ordered class-index pairs used by sklearn one-vs-one fitting."""
    return np.asarray([(i, j) for i in range(n_classes) for j in range(i + 1, n_classes)], dtype=np.int64)


@register_atom(witness_output_code_book)
@icontract.require(lambda n_classes, code_size, random_state: _code_book_params_valid(n_classes, code_size, random_state), "code book dimensions and random_state must be valid")
@icontract.ensure(lambda result, n_classes, code_size, estimator_has_decision_function: _code_book_valid(result, n_classes, code_size, estimator_has_decision_function), "code book must contain sklearn output-code values")
def output_code_book(
    n_classes: int,
    *,
    code_size: float = 1.5,
    random_state: int = 0,
    estimator_has_decision_function: bool = True,
) -> NDArray[np.float64]:
    """Generate sklearn's deterministic random output-code book for class rows."""
    rng = np.random.RandomState(random_state)
    n_estimators = int(n_classes * float(code_size))
    code_book = rng.uniform(size=(n_classes, n_estimators))
    code_book[code_book > 0.5] = 1.0
    if estimator_has_decision_function:
        code_book[code_book != 1.0] = -1.0
    else:
        code_book[code_book != 1.0] = 0.0
    return np.asarray(code_book, dtype=np.float64)


@register_atom(witness_output_code_decode)
@icontract.require(lambda responses, code_book, classes: _decode_inputs_valid(responses, code_book, classes), "responses, code_book, and classes must have compatible dimensions")
@icontract.ensure(lambda result, responses, classes: _labels_valid(result, np.asarray(responses).shape[0], classes), "decoded labels must come from the class vector")
def output_code_decode(
    responses: NDArray[np.float64],
    code_book: NDArray[np.float64],
    classes: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Decode output-code responses by nearest Euclidean code-book row."""
    response_values = np.asarray(responses, dtype=np.float64)
    code_values = np.asarray(code_book, dtype=np.float64)
    class_values = np.asarray(classes, dtype=np.float64)
    distances = np.sum((response_values[:, np.newaxis, :] - code_values[np.newaxis, :, :]) ** 2, axis=2)
    return np.asarray(class_values[np.argmin(distances, axis=1)], dtype=np.float64)
