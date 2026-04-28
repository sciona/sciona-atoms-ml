"""Output-code matrix preparation helpers adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_output_code_fit_classes,
    witness_output_code_response_matrix,
    witness_output_code_target_matrix,
)


def _nonempty_targets(y: NDArray[np.float64]) -> bool:
    values = np.asarray(y, dtype=np.float64)
    return bool(values.ndim == 1 and values.shape[0] >= 1 and np.all(np.isfinite(values)))


def _classes_valid(classes: NDArray[np.float64]) -> bool:
    values = np.asarray(classes, dtype=np.float64)
    return bool(
        values.ndim == 1
        and values.shape[0] >= 1
        and np.all(np.isfinite(values))
        and np.unique(values).shape[0] == values.shape[0]
    )


def _target_matrix_inputs_valid(
    y: NDArray[np.float64],
    classes: NDArray[np.float64],
    code_book: NDArray[np.float64],
) -> bool:
    target_values = np.asarray(y, dtype=np.float64)
    class_values = np.asarray(classes, dtype=np.float64)
    code_values = np.asarray(code_book, dtype=np.float64)
    return bool(
        _nonempty_targets(target_values)
        and _classes_valid(class_values)
        and code_values.ndim == 2
        and code_values.shape[0] == class_values.shape[0]
        and code_values.shape[1] >= 1
        and np.all(np.isfinite(code_values))
        and np.setdiff1d(np.unique(target_values), class_values).size == 0
    )


def _target_matrix_valid(
    result: NDArray[np.int64],
    y: NDArray[np.float64],
    code_book: NDArray[np.float64],
) -> bool:
    values = np.asarray(result)
    target_values = np.asarray(y, dtype=np.float64)
    code_values = np.asarray(code_book, dtype=np.float64)
    return bool(
        values.shape == (target_values.shape[0], code_values.shape[1])
        and np.issubdtype(values.dtype, np.integer)
        and np.all(np.isin(values, np.array([-1, 0, 1], dtype=np.int64)))
    )


def _prediction_matrix_valid(estimator_predictions: NDArray[np.float64]) -> bool:
    values = np.asarray(estimator_predictions, dtype=np.float64)
    return bool(values.ndim == 2 and values.shape[0] >= 1 and values.shape[1] >= 1 and np.all(np.isfinite(values)))


def _response_matrix_valid(
    result: NDArray[np.float64],
    estimator_predictions: NDArray[np.float64],
) -> bool:
    values = np.asarray(result, dtype=np.float64)
    prediction_values = np.asarray(estimator_predictions, dtype=np.float64)
    return bool(
        values.shape == (prediction_values.shape[1], prediction_values.shape[0])
        and np.all(np.isfinite(values))
        and values.flags.c_contiguous
    )


@register_atom(witness_output_code_fit_classes)
@icontract.require(lambda y: _nonempty_targets(y), "y must be a nonempty finite 1D target vector")
@icontract.ensure(lambda result: _classes_valid(result), "classes must be a finite unique class vector")
def output_code_fit_classes(y: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return sklearn's sorted unique class vector for output-code fitting."""
    return np.asarray(np.unique(np.asarray(y, dtype=np.float64)), dtype=np.float64)


@register_atom(witness_output_code_target_matrix)
@icontract.require(
    lambda y, classes, code_book: _target_matrix_inputs_valid(y, classes, code_book),
    "y, classes, and code_book must be compatible and y must only contain known classes",
)
@icontract.ensure(
    lambda result, y, code_book: _target_matrix_valid(result, y, code_book),
    "target matrix must be an integer sample-by-estimator code matrix",
)
def output_code_target_matrix(
    y: NDArray[np.float64],
    classes: NDArray[np.float64],
    code_book: NDArray[np.float64],
) -> NDArray[np.int64]:
    """Gather one output-code row per target label as sklearn's integer fit matrix."""
    target_values = np.asarray(y, dtype=np.float64)
    class_values = np.asarray(classes, dtype=np.float64)
    code_values = np.asarray(code_book, dtype=np.float64)
    classes_index = {class_value: index for index, class_value in enumerate(class_values)}
    return np.asarray(
        [code_values[classes_index[target_values[i]]] for i in range(target_values.shape[0])],
        dtype=np.int64,
    )


@register_atom(witness_output_code_response_matrix)
@icontract.require(
    lambda estimator_predictions: _prediction_matrix_valid(estimator_predictions),
    "estimator_predictions must be a finite estimator-by-sample matrix",
)
@icontract.ensure(
    lambda result, estimator_predictions: _response_matrix_valid(result, estimator_predictions),
    "response matrix must be a finite C-contiguous sample-by-estimator matrix",
)
def output_code_response_matrix(
    estimator_predictions: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Transpose estimator-major binary responses into sklearn's sample-major decode matrix."""
    return np.array(np.asarray(estimator_predictions, dtype=np.float64), order="F", dtype=np.float64).T
