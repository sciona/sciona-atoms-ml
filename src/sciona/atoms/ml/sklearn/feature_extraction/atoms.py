"""Dictionary feature extraction atoms adapted from scikit-learn."""

from __future__ import annotations

from collections.abc import Mapping

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .state_models import DictVectorizerState
from .witnesses import (
    witness_dict_vectorizer_feature_names,
    witness_dict_vectorizer_fit,
    witness_dict_vectorizer_inverse_transform,
    witness_dict_vectorizer_restrict,
    witness_dict_vectorizer_transform,
)

FeatureValue = int | float | str | tuple[str, ...]
FeatureRecords = tuple[dict[str, FeatureValue], ...]


def _records_valid(records: FeatureRecords) -> bool:
    if not isinstance(records, tuple) or len(records) == 0:
        return False
    for record in records:
        if not isinstance(record, dict):
            return False
        for key, value in record.items():
            if not isinstance(key, str):
                return False
            if isinstance(value, (int, float, str)):
                continue
            if isinstance(value, tuple) and all(isinstance(item, str) for item in value):
                continue
            return False
    return True


def _state_valid(state: DictVectorizerState) -> bool:
    return bool(
        state.separator
        and len(state.feature_names) == len(state.vocabulary)
        and set(state.feature_names) == set(state.vocabulary)
        and sorted(state.vocabulary.values()) == list(range(len(state.feature_names)))
        and all(state.feature_names[index] == name for name, index in state.vocabulary.items())
    )


def _matrix_2d(X: NDArray[np.float64]) -> bool:
    try:
        values = np.asarray(X, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(values.ndim == 2)


def _fit_result_valid(result: DictVectorizerState) -> bool:
    return _state_valid(result)


def _transform_result_valid(result: NDArray[np.float64], records: FeatureRecords, state: DictVectorizerState) -> bool:
    values = np.asarray(result, dtype=np.float64)
    return bool(values.shape == (len(records), len(state.feature_names)) and np.all(np.isfinite(values)))


def _inverse_result_valid(result: list[dict[str, float]], X: NDArray[np.float64]) -> bool:
    return bool(len(result) == np.asarray(X).shape[0] and all(isinstance(record, dict) for record in result))


def _support_valid(support: tuple[int, ...], state: DictVectorizerState) -> bool:
    return bool(
        isinstance(support, tuple)
        and len(set(support)) == len(support)
        and all(isinstance(index, int) and 0 <= index < len(state.feature_names) for index in support)
    )


def _feature_name(key: str, value: FeatureValue, separator: str) -> str | None:
    if isinstance(value, str):
        return f"{key}{separator}{value}"
    if isinstance(value, (int, float)):
        return key
    return None


def _iterable_feature_names(key: str, values: tuple[str, ...], separator: str) -> tuple[str, ...]:
    return tuple(f"{key}{separator}{value}" for value in values)


@register_atom(witness_dict_vectorizer_fit)
@icontract.require(lambda records: _records_valid(records), "records must be non-empty dictionaries with supported feature values")
@icontract.require(lambda separator: isinstance(separator, str) and len(separator) > 0, "separator must be a non-empty string")
@icontract.ensure(lambda result: _fit_result_valid(result), "state must contain a valid feature vocabulary")
def dict_vectorizer_fit(
    records: FeatureRecords,
    *,
    separator: str = "=",
    sort: bool = True,
) -> DictVectorizerState:
    """Learn feature names and indices from dense dictionary records."""
    feature_names: list[str] = []
    vocabulary: dict[str, int] = {}

    for record in records:
        for key, value in record.items():
            if isinstance(value, tuple):
                names = _iterable_feature_names(key, value, separator)
            else:
                feature_name = _feature_name(key, value, separator)
                names = () if feature_name is None else (feature_name,)
            for name in names:
                if name not in vocabulary:
                    vocabulary[name] = len(feature_names)
                    feature_names.append(name)

    if sort:
        feature_names.sort()
        vocabulary = {name: index for index, name in enumerate(feature_names)}

    return DictVectorizerState(
        feature_names=tuple(feature_names),
        vocabulary=vocabulary,
        separator=separator,
    )


@register_atom(witness_dict_vectorizer_transform)
@icontract.require(lambda records: _records_valid(records), "records must be non-empty dictionaries with supported feature values")
@icontract.require(lambda state: _state_valid(state), "state must contain a valid feature vocabulary")
@icontract.ensure(lambda result, records, state: _transform_result_valid(result, records, state), "matrix must match records and fitted features")
def dict_vectorizer_transform(records: FeatureRecords, state: DictVectorizerState) -> NDArray[np.float64]:
    """Transform dictionary records into a dense numeric feature matrix."""
    matrix = np.zeros((len(records), len(state.feature_names)), dtype=np.float64)
    for row, record in enumerate(records):
        for key, value in record.items():
            if isinstance(value, str):
                name = f"{key}{state.separator}{value}"
                if name in state.vocabulary:
                    matrix[row, state.vocabulary[name]] += 1.0
            elif isinstance(value, tuple):
                for name in _iterable_feature_names(key, value, state.separator):
                    if name in state.vocabulary:
                        matrix[row, state.vocabulary[name]] += 1.0
            elif key in state.vocabulary:
                matrix[row, state.vocabulary[key]] += float(value)
    return matrix


@register_atom(witness_dict_vectorizer_inverse_transform)
@icontract.require(lambda X: _matrix_2d(X), "X must be a dense numeric 2D matrix")
@icontract.require(lambda state: _state_valid(state), "state must contain a valid feature vocabulary")
@icontract.require(lambda X, state: np.asarray(X).shape[1] == len(state.feature_names), "X feature count must match fitted state")
@icontract.ensure(lambda result, X: _inverse_result_valid(result, X), "inverse records must match sample count")
def dict_vectorizer_inverse_transform(X: NDArray[np.float64], state: DictVectorizerState) -> list[dict[str, float]]:
    """Convert nonzero dense feature matrix entries back to feature mappings."""
    values = np.asarray(X, dtype=np.float64)
    records: list[dict[str, float]] = []
    for row in range(values.shape[0]):
        record: dict[str, float] = {}
        for col, value in enumerate(values[row, :]):
            if value != 0.0:
                record[state.feature_names[col]] = float(value)
        records.append(record)
    return records


@register_atom(witness_dict_vectorizer_feature_names)
@icontract.require(lambda state: _state_valid(state), "state must contain a valid feature vocabulary")
@icontract.ensure(lambda result, state: len(result) == len(state.feature_names), "feature names must match fitted state")
def dict_vectorizer_feature_names(state: DictVectorizerState) -> tuple[str, ...]:
    """Return learned feature names in output-column order."""
    return state.feature_names


@register_atom(witness_dict_vectorizer_restrict)
@icontract.require(lambda state: _state_valid(state), "state must contain a valid feature vocabulary")
@icontract.require(lambda support, state: _support_valid(support, state), "support indices must be unique valid feature columns")
@icontract.ensure(lambda result, support: len(result.feature_names) == len(support), "restricted state must match support size")
@icontract.ensure(lambda result: _state_valid(result), "restricted state must contain a valid feature vocabulary")
def dict_vectorizer_restrict(state: DictVectorizerState, support: tuple[int, ...]) -> DictVectorizerState:
    """Return a vocabulary state restricted to selected feature columns."""
    feature_names = tuple(state.feature_names[index] for index in support)
    vocabulary = {name: index for index, name in enumerate(feature_names)}
    return DictVectorizerState(
        feature_names=feature_names,
        vocabulary=vocabulary,
        separator=state.separator,
    )
