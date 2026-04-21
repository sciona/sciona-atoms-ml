"""Dictionary feature extraction atoms adapted from scikit-learn."""

from __future__ import annotations

from collections.abc import Mapping

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .state_models import DictVectorizerState, TfidfTransformerState
from .witnesses import (
    witness_dict_vectorizer_feature_names,
    witness_dict_vectorizer_fit,
    witness_dict_vectorizer_inverse_transform,
    witness_dict_vectorizer_restrict,
    witness_dict_vectorizer_transform,
    witness_tfidf_document_frequency,
    witness_tfidf_idf,
    witness_tfidf_transform,
    witness_tfidf_transformer_fit,
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


def _nonnegative_count_matrix(X: NDArray[np.float64]) -> bool:
    try:
        values = np.asarray(X, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(values.ndim == 2 and values.shape[0] >= 1 and values.shape[1] >= 1 and np.all(np.isfinite(values)) and np.all(values >= 0.0))


def _norm_valid(norm: str | None) -> bool:
    return norm in {"l1", "l2", None}


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


def _document_frequency_result_valid(result: NDArray[np.float64], X: NDArray[np.float64]) -> bool:
    values = np.asarray(result, dtype=np.float64)
    return bool(values.shape == (np.asarray(X).shape[1],) and np.all(np.isfinite(values)) and np.all(values >= 0.0) and np.all(values <= np.asarray(X).shape[0]))


def _idf_inputs_valid(document_frequency: NDArray[np.float64], n_samples: int) -> bool:
    values = np.asarray(document_frequency, dtype=np.float64)
    return bool(values.ndim == 1 and values.shape[0] >= 1 and n_samples >= 1 and np.all(np.isfinite(values)) and np.all(values >= 0.0) and np.all(values <= n_samples))


def _idf_result_valid(result: NDArray[np.float64], document_frequency: NDArray[np.float64]) -> bool:
    values = np.asarray(result, dtype=np.float64)
    return bool(values.shape == np.asarray(document_frequency).shape and np.all(np.isfinite(values)) and np.all(values >= 1.0))


def _tfidf_state_valid(state: TfidfTransformerState) -> bool:
    idf_valid = state.idf is None or (
        np.asarray(state.idf, dtype=np.float64).shape == (state.n_features_in,)
        and np.all(np.isfinite(state.idf))
        and np.all(state.idf >= 1.0)
    )
    return bool(
        state.n_features_in >= 1
        and _norm_valid(state.norm)
        and isinstance(state.use_idf, bool)
        and isinstance(state.smooth_idf, bool)
        and isinstance(state.sublinear_tf, bool)
        and ((state.use_idf and state.idf is not None) or (not state.use_idf and state.idf is None))
        and idf_valid
    )


def _tfidf_fit_result_valid(result: TfidfTransformerState, X: NDArray[np.float64]) -> bool:
    return bool(_tfidf_state_valid(result) and result.n_features_in == np.asarray(X).shape[1])


def _tfidf_transform_inputs_valid(X: NDArray[np.float64], state: TfidfTransformerState) -> bool:
    return bool(_nonnegative_count_matrix(X) and _tfidf_state_valid(state) and np.asarray(X).shape[1] == state.n_features_in)


def _tfidf_transform_result_valid(result: NDArray[np.float64], X: NDArray[np.float64]) -> bool:
    values = np.asarray(result, dtype=np.float64)
    return bool(values.shape == np.asarray(X).shape and np.all(np.isfinite(values)) and np.all(values >= 0.0))


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


@register_atom(witness_tfidf_document_frequency)
@icontract.require(lambda X: _nonnegative_count_matrix(X), "X must be a dense finite nonnegative count matrix")
@icontract.ensure(lambda result, X: _document_frequency_result_valid(result, X), "document frequencies must match feature count")
def tfidf_document_frequency(X: NDArray[np.float64]) -> NDArray[np.float64]:
    """Count how many documents contain each term feature."""
    values = np.asarray(X, dtype=np.float64)
    return np.count_nonzero(values > 0.0, axis=0).astype(np.float64)


@register_atom(witness_tfidf_idf)
@icontract.require(lambda document_frequency, n_samples: _idf_inputs_valid(document_frequency, n_samples), "document frequencies must be valid for n_samples")
@icontract.ensure(lambda result, document_frequency: _idf_result_valid(result, document_frequency), "idf values must be finite positive feature weights")
def tfidf_idf(
    document_frequency: NDArray[np.float64],
    n_samples: int,
    *,
    smooth_idf: bool = True,
) -> NDArray[np.float64]:
    """Compute inverse-document-frequency weights from document frequencies."""
    df = np.asarray(document_frequency, dtype=np.float64).copy()
    sample_count = int(n_samples)
    if smooth_idf:
        df += 1.0
        sample_count += 1
    return np.log(np.full_like(df, fill_value=sample_count, dtype=np.float64) / df) + 1.0


@register_atom(witness_tfidf_transformer_fit)
@icontract.require(lambda X: _nonnegative_count_matrix(X), "X must be a dense finite nonnegative count matrix")
@icontract.require(lambda norm: _norm_valid(norm), "norm must be 'l1', 'l2', or None")
@icontract.ensure(lambda result, X: _tfidf_fit_result_valid(result, X), "state must describe a fitted dense TF-IDF transformer")
def tfidf_transformer_fit(
    X: NDArray[np.float64],
    *,
    norm: str | None = "l2",
    use_idf: bool = True,
    smooth_idf: bool = True,
    sublinear_tf: bool = False,
) -> TfidfTransformerState:
    """Fit dense TF-IDF transformer weights for a count matrix."""
    idf = tfidf_idf(tfidf_document_frequency(X), np.asarray(X).shape[0], smooth_idf=smooth_idf) if use_idf else None
    return TfidfTransformerState(
        idf=idf,
        norm=norm,
        use_idf=bool(use_idf),
        smooth_idf=bool(smooth_idf),
        sublinear_tf=bool(sublinear_tf),
        n_features_in=int(np.asarray(X).shape[1]),
    )


@register_atom(witness_tfidf_transform)
@icontract.require(lambda X, state: _tfidf_transform_inputs_valid(X, state), "X must match a valid fitted TF-IDF state")
@icontract.ensure(lambda result, X: _tfidf_transform_result_valid(result, X), "transformed matrix must preserve shape and nonnegative finite weights")
def tfidf_transform(X: NDArray[np.float64], state: TfidfTransformerState) -> NDArray[np.float64]:
    """Transform dense term counts into TF-IDF weights."""
    values = np.asarray(X, dtype=np.float64).copy()
    if state.sublinear_tf:
        mask = values > 0.0
        values[mask] = np.log(values[mask]) + 1.0
    if state.idf is not None:
        values *= state.idf[np.newaxis, :]
    if state.norm == "l1":
        row_norm = np.sum(np.abs(values), axis=1)
        nonzero = row_norm > 0.0
        values[nonzero] /= row_norm[nonzero, np.newaxis]
    elif state.norm == "l2":
        row_norm = np.sqrt(np.sum(values * values, axis=1))
        nonzero = row_norm > 0.0
        values[nonzero] /= row_norm[nonzero, np.newaxis]
    return values
