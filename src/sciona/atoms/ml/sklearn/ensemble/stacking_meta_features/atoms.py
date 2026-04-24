"""Stacking meta-feature helper atoms adapted from scikit-learn."""

from __future__ import annotations

from collections.abc import Sequence

import icontract
import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_stacking_feature_names_out,
    witness_stacking_meta_feature_matrix,
    witness_stacking_meta_feature_widths,
)

PredictionEntry = NDArray[np.float64] | tuple[NDArray[np.float64], ...] | list[NDArray[np.float64]]
MatrixLike = NDArray[np.float64] | sp.spmatrix


def _finite_dense_matrix(values: object) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 2 and array.shape[0] >= 1 and array.shape[1] >= 1 and np.all(np.isfinite(array)))


def _finite_matrix_like(values: object) -> bool:
    if sp.issparse(values):
        matrix = values
        return bool(matrix.ndim == 2 and matrix.shape[0] >= 1 and matrix.shape[1] >= 1 and np.all(np.isfinite(matrix.data)))
    return _finite_dense_matrix(values)


def _nonempty_string_tuple(values: tuple[str, ...]) -> bool:
    return bool(
        isinstance(values, tuple)
        and len(values) >= 1
        and all(isinstance(value, str) and value in {"predict", "predict_proba", "decision_function"} for value in values)
    )


def _prediction_entry_sample_count(entry: PredictionEntry) -> int:
    if isinstance(entry, (list, tuple)):
        return int(np.asarray(entry[0], dtype=np.float64).shape[0])
    return int(np.asarray(entry, dtype=np.float64).shape[0])


def _prediction_entry_valid(entry: PredictionEntry, method: str, is_binary_classification: bool) -> bool:
    if isinstance(entry, (list, tuple)):
        if len(entry) < 1 or method != "predict_proba":
            return False
        n_samples: int | None = None
        for block in entry:
            if not _finite_dense_matrix(block):
                return False
            values = np.asarray(block, dtype=np.float64)
            if values.shape[1] < 2:
                return False
            if n_samples is None:
                n_samples = int(values.shape[0])
            elif int(values.shape[0]) != n_samples:
                return False
        return True

    try:
        values = np.asarray(entry, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    if values.ndim == 1:
        return bool(values.shape[0] >= 1 and np.all(np.isfinite(values)))
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 1 or not np.all(np.isfinite(values)):
        return False
    if method == "predict_proba" and is_binary_classification and values.shape[1] < 2:
        return False
    return True


def _prediction_entries_valid(
    predictions: tuple[PredictionEntry, ...],
    stack_method_names: tuple[str, ...],
    is_binary_classification: bool,
) -> bool:
    if not (
        isinstance(predictions, tuple)
        and len(predictions) >= 1
        and _nonempty_string_tuple(stack_method_names)
        and len(predictions) == len(stack_method_names)
    ):
        return False
    n_samples: int | None = None
    for entry, method in zip(predictions, stack_method_names):
        if not _prediction_entry_valid(entry, method, is_binary_classification):
            return False
        rows = _prediction_entry_sample_count(entry)
        if n_samples is None:
            n_samples = rows
        elif rows != n_samples:
            return False
    return True


def _passthrough_inputs_valid(
    predictions: tuple[PredictionEntry, ...],
    stack_method_names: tuple[str, ...],
    is_binary_classification: bool,
    X: MatrixLike | None,
    passthrough: bool,
) -> bool:
    if not _prediction_entries_valid(predictions, stack_method_names, is_binary_classification):
        return False
    if not passthrough:
        return True
    if X is None or not _finite_matrix_like(X):
        return False
    return bool(_prediction_entry_sample_count(predictions[0]) == int(X.shape[0]))


def _normalized_prediction_blocks(
    predictions: tuple[PredictionEntry, ...],
    stack_method_names: tuple[str, ...],
    is_binary_classification: bool,
) -> tuple[NDArray[np.float64], ...]:
    blocks: list[NDArray[np.float64]] = []
    for method, entry in zip(stack_method_names, predictions):
        if isinstance(entry, (list, tuple)):
            for block in entry:
                values = np.asarray(block, dtype=np.float64)
                blocks.append(np.asarray(values[:, 1:], dtype=np.float64))
            continue

        values = np.asarray(entry, dtype=np.float64)
        if values.ndim == 1:
            blocks.append(np.asarray(values.reshape(-1, 1), dtype=np.float64))
        elif method == "predict_proba" and is_binary_classification:
            blocks.append(np.asarray(values[:, 1:], dtype=np.float64))
        else:
            blocks.append(np.asarray(values, dtype=np.float64))
    return tuple(blocks)


def _feature_widths(blocks: tuple[NDArray[np.float64], ...]) -> NDArray[np.int64]:
    return np.asarray([int(block.shape[1]) for block in blocks], dtype=np.int64)


def _meta_feature_matrix_valid(
    result: MatrixLike,
    predictions: tuple[PredictionEntry, ...],
    stack_method_names: tuple[str, ...],
    is_binary_classification: bool,
    X: MatrixLike | None,
    passthrough: bool,
) -> bool:
    if sp.issparse(result):
        if not passthrough or X is None or not sp.issparse(X):
            return False
        rows = int(result.shape[0])
        cols = int(result.shape[1])
        return bool(
            rows == _prediction_entry_sample_count(predictions[0])
            and cols == int(np.sum(_feature_widths(_normalized_prediction_blocks(predictions, stack_method_names, is_binary_classification)))) + int(X.shape[1])
            and result.getformat() == X.getformat()
            and np.all(np.isfinite(result.data))
        )

    try:
        values = np.asarray(result, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    if values.ndim != 2 or not np.all(np.isfinite(values)):
        return False
    expected_cols = int(
        np.sum(_feature_widths(_normalized_prediction_blocks(predictions, stack_method_names, is_binary_classification)))
    )
    if passthrough and X is not None:
        expected_cols += int(X.shape[1])
    return bool(values.shape == (_prediction_entry_sample_count(predictions[0]), expected_cols))


def _feature_widths_valid(
    result: NDArray[np.int64],
    predictions: tuple[PredictionEntry, ...],
    stack_method_names: tuple[str, ...],
    is_binary_classification: bool,
) -> bool:
    try:
        values = np.asarray(result, dtype=np.int64)
    except (TypeError, ValueError):
        return False
    expected = _feature_widths(_normalized_prediction_blocks(predictions, stack_method_names, is_binary_classification))
    return bool(values.shape == expected.shape and np.array_equal(values, expected))


def _feature_name_inputs_valid(
    class_name: str,
    estimator_names: tuple[str, ...],
    feature_widths: NDArray[np.int64],
    input_features: Sequence[str] | NDArray[np.object_] | None,
    passthrough: bool,
) -> bool:
    try:
        widths = np.asarray(feature_widths, dtype=np.int64)
    except (TypeError, ValueError):
        return False
    if not (
        isinstance(class_name, str)
        and len(class_name) >= 1
        and isinstance(estimator_names, tuple)
        and len(estimator_names) >= 1
        and all(isinstance(name, str) and len(name) >= 1 for name in estimator_names)
        and widths.ndim == 1
        and widths.shape[0] == len(estimator_names)
        and np.all(widths >= 1)
    ):
        return False
    if not passthrough:
        return True
    if input_features is None:
        return False
    if isinstance(input_features, np.ndarray):
        if input_features.ndim != 1 or input_features.shape[0] < 1:
            return False
        return bool(all(isinstance(item, str) and len(item) >= 1 for item in input_features.tolist()))
    if not isinstance(input_features, Sequence) or len(input_features) < 1:
        return False
    return bool(all(isinstance(item, str) and len(item) >= 1 for item in input_features))


def _feature_names_out_valid(
    result: NDArray[np.object_],
    estimator_names: tuple[str, ...],
    feature_widths: NDArray[np.int64],
    input_features: Sequence[str] | NDArray[np.object_] | None,
    passthrough: bool,
) -> bool:
    try:
        values = np.asarray(result, dtype=object)
        widths = np.asarray(feature_widths, dtype=np.int64)
    except (TypeError, ValueError):
        return False
    expected_length = int(np.sum(widths)) + (0 if not passthrough or input_features is None else len(input_features))
    return bool(values.shape == (expected_length,) and all(isinstance(item, str) and len(item) >= 1 for item in values.tolist()))


@register_atom(witness_stacking_meta_feature_matrix)
@icontract.require(
    lambda predictions, stack_method_names, is_binary_classification, X=None, passthrough=False: _passthrough_inputs_valid(
        predictions, stack_method_names, is_binary_classification, X, passthrough
    ),
    "predictions must be nonempty, aligned, finite sklearn response outputs, and X must be a matching finite matrix when passthrough is enabled",
)
@icontract.ensure(
    lambda result, predictions, stack_method_names, is_binary_classification, X=None, passthrough=False: _meta_feature_matrix_valid(
        result, predictions, stack_method_names, is_binary_classification, X, passthrough
    ),
    "meta-feature matrix must preserve the shared sample count and expected concatenated width",
)
def stacking_meta_feature_matrix(
    predictions: tuple[PredictionEntry, ...],
    stack_method_names: tuple[str, ...],
    *,
    is_binary_classification: bool,
    X: MatrixLike | None = None,
    passthrough: bool = False,
) -> MatrixLike:
    """Construct stacking meta-features from supplied estimator prediction outputs."""
    blocks = _normalized_prediction_blocks(predictions, stack_method_names, is_binary_classification)
    if passthrough:
        assert X is not None
        if sp.issparse(X):
            return sp.hstack((*blocks, X), format=X.getformat())
        return np.hstack((*blocks, np.asarray(X, dtype=np.float64)))
    return np.hstack(blocks)


@register_atom(witness_stacking_meta_feature_widths)
@icontract.require(
    lambda predictions, stack_method_names, is_binary_classification: _prediction_entries_valid(
        predictions, stack_method_names, is_binary_classification
    ),
    "predictions must be nonempty, aligned, and finite sklearn response outputs",
)
@icontract.ensure(
    lambda result, predictions, stack_method_names, is_binary_classification: _feature_widths_valid(
        result, predictions, stack_method_names, is_binary_classification
    ),
    "feature widths must match the widths of sklearn-style normalized stacking prediction blocks",
)
def stacking_meta_feature_widths(
    predictions: tuple[PredictionEntry, ...],
    stack_method_names: tuple[str, ...],
    *,
    is_binary_classification: bool,
) -> NDArray[np.int64]:
    """Return stacking meta-feature widths after sklearn-style prediction normalization."""
    return _feature_widths(_normalized_prediction_blocks(predictions, stack_method_names, is_binary_classification))


@register_atom(witness_stacking_feature_names_out)
@icontract.require(
    lambda class_name, estimator_names, feature_widths, input_features=None, passthrough=False: _feature_name_inputs_valid(
        class_name, estimator_names, feature_widths, input_features, passthrough
    ),
    "class_name, estimator_names, and feature_widths must be aligned; passthrough names require a nonempty input feature sequence",
)
@icontract.ensure(
    lambda result, estimator_names, feature_widths, input_features=None, passthrough=False: _feature_names_out_valid(
        result, estimator_names, feature_widths, input_features, passthrough
    ),
    "feature names must match the expanded width count and optional passthrough feature count",
)
def stacking_feature_names_out(
    class_name: str,
    estimator_names: tuple[str, ...],
    feature_widths: NDArray[np.int64],
    *,
    input_features: Sequence[str] | NDArray[np.object_] | None = None,
    passthrough: bool = False,
) -> NDArray[np.object_]:
    """Build sklearn-style stacking transform feature names from estimator names and output widths."""
    widths = np.asarray(feature_widths, dtype=np.int64)
    meta_names: list[str] = []
    for estimator_name, width in zip(estimator_names, widths):
        if int(width) == 1:
            meta_names.append(f"{class_name}_{estimator_name}")
        else:
            meta_names.extend(f"{class_name}_{estimator_name}{index}" for index in range(int(width)))

    if passthrough:
        assert input_features is not None
        return np.asarray(meta_names + list(input_features), dtype=object)
    return np.asarray(meta_names, dtype=object)
