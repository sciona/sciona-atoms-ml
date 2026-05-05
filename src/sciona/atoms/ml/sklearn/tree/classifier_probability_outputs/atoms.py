"""Sklearn tree classifier probability-output atoms adapted from scikit-learn."""

from __future__ import annotations

from collections.abc import Sequence

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_tree_predict_log_proba_multioutput,
    witness_tree_predict_log_proba_single_output,
    witness_tree_predict_proba_multioutput,
    witness_tree_predict_proba_single_output,
)


def _probability_matrix(values: object) -> bool:
    try:
        matrix = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(
        matrix.ndim == 2
        and matrix.shape[0] >= 1
        and matrix.shape[1] >= 1
        and np.all(np.isfinite(matrix))
        and np.all(matrix >= 0.0)
        and np.all(matrix <= 1.0)
        and np.allclose(matrix.sum(axis=1), 1.0)
    )


def _probability_tensor(values: object) -> bool:
    try:
        tensor = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(
        tensor.ndim == 3
        and tensor.shape[0] >= 1
        and tensor.shape[1] >= 1
        and tensor.shape[2] >= 1
        and np.all(np.isfinite(tensor))
        and np.all(tensor >= 0.0)
        and np.all(tensor <= 1.0)
    )


def _class_count(value: object) -> bool:
    return isinstance(value, (int, np.integer)) and int(value) >= 1


def _class_counts(values: object) -> bool:
    return bool(
        isinstance(values, tuple)
        and len(values) >= 1
        and all(_class_count(value) for value in values)
    )


def _multioutput_probability_inputs(probabilities: object, n_classes: object) -> bool:
    if not (_probability_tensor(probabilities) and _class_counts(n_classes)):
        return False
    tensor = np.asarray(probabilities, dtype=np.float64)
    counts = tuple(int(value) for value in n_classes)
    return bool(tensor.shape[1] == len(counts) and all(tensor.shape[2] >= count for count in counts))


def _probability_matrix_sequence(values: object) -> bool:
    return bool(
        isinstance(values, Sequence)
        and len(values) >= 1
        and all(_probability_matrix(matrix) for matrix in values)
    )


def _log_probability_matrix(values: object) -> bool:
    try:
        matrix = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(matrix.ndim == 2 and matrix.shape[0] >= 1 and matrix.shape[1] >= 1 and np.all(matrix <= 0.0))


def _log_probability_matrix_sequence(values: object) -> bool:
    return bool(
        isinstance(values, list)
        and len(values) >= 1
        and all(_log_probability_matrix(matrix) for matrix in values)
    )


def _logged_probabilities(values: object) -> NDArray[np.float64]:
    with np.errstate(divide="ignore"):
        return np.log(np.asarray(values, dtype=np.float64))


@register_atom(witness_tree_predict_proba_single_output)
@icontract.require(
    lambda probabilities, n_classes: _probability_matrix(probabilities) and _class_count(n_classes),
    "probabilities must be a nonempty sample-by-class probability matrix and n_classes must be positive",
)
@icontract.require(
    lambda probabilities, n_classes: np.asarray(probabilities, dtype=np.float64).shape[1] >= int(n_classes),
    "probabilities must have at least n_classes columns",
)
@icontract.ensure(
    lambda result, probabilities, n_classes: _probability_matrix(result)
    and np.allclose(
        np.asarray(result, dtype=np.float64),
        np.asarray(probabilities, dtype=np.float64)[:, : int(n_classes)],
    ),
    "single-output probabilities must equal the leading class slice",
)
def tree_predict_proba_single_output(
    probabilities: NDArray[np.float64],
    n_classes: int,
) -> NDArray[np.float64]:
    """Return the single-output DecisionTreeClassifier.predict_proba slice."""
    matrix = np.asarray(probabilities, dtype=np.float64)
    return np.asarray(matrix[:, : int(n_classes)], dtype=np.float64)


@register_atom(witness_tree_predict_proba_multioutput)
@icontract.require(
    lambda probabilities, n_classes: _multioutput_probability_inputs(probabilities, n_classes),
    "probabilities must be a nonempty sample-by-output-by-class tensor aligned with n_classes",
)
@icontract.ensure(
    lambda result, probabilities, n_classes: isinstance(result, list)
    and len(result) == len(n_classes)
    and all(
        _probability_matrix(block)
        and np.allclose(
            np.asarray(block, dtype=np.float64),
            np.asarray(probabilities, dtype=np.float64)[:, index, : int(n_classes[index])],
        )
        for index, block in enumerate(result)
    ),
    "multioutput probabilities must equal the per-output leading class slices",
)
def tree_predict_proba_multioutput(
    probabilities: NDArray[np.float64],
    n_classes: tuple[int, ...],
) -> list[NDArray[np.float64]]:
    """Return the multioutput DecisionTreeClassifier.predict_proba slices."""
    tensor = np.asarray(probabilities, dtype=np.float64)
    return [
        np.asarray(tensor[:, index, : int(class_count)], dtype=np.float64)
        for index, class_count in enumerate(n_classes)
    ]


@register_atom(witness_tree_predict_log_proba_single_output)
@icontract.require(
    lambda probabilities: _probability_matrix(probabilities),
    "probabilities must be a nonempty sample-by-class probability matrix",
)
@icontract.ensure(
    lambda result, probabilities: _log_probability_matrix(result)
    and np.allclose(
        np.asarray(result, dtype=np.float64),
        _logged_probabilities(probabilities),
        equal_nan=True,
    ),
    "single-output log probabilities must equal the elementwise log",
)
def tree_predict_log_proba_single_output(
    probabilities: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return single-output DecisionTreeClassifier.predict_log_proba values."""
    matrix = np.asarray(probabilities, dtype=np.float64)
    with np.errstate(divide="ignore"):
        return np.log(matrix)


@register_atom(witness_tree_predict_log_proba_multioutput)
@icontract.require(
    lambda probabilities: _probability_matrix_sequence(probabilities),
    "probabilities must be a nonempty sequence of probability matrices",
)
@icontract.ensure(
    lambda result, probabilities: _log_probability_matrix_sequence(result)
    and len(result) == len(probabilities)
    and all(
        np.allclose(
            np.asarray(result[index], dtype=np.float64),
            _logged_probabilities(probabilities[index]),
            equal_nan=True,
        )
        for index in range(len(probabilities))
    ),
    "multioutput log probabilities must equal the per-output elementwise log",
)
def tree_predict_log_proba_multioutput(
    probabilities: tuple[NDArray[np.float64], ...],
) -> list[NDArray[np.float64]]:
    """Return multioutput DecisionTreeClassifier.predict_log_proba values."""
    outputs: list[NDArray[np.float64]] = []
    with np.errstate(divide="ignore"):
        for block in probabilities:
            outputs.append(np.log(np.asarray(block, dtype=np.float64)))
    return outputs
