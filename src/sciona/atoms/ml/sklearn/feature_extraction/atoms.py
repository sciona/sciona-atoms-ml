"""Dictionary feature extraction atoms adapted from scikit-learn."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .state_models import CountVectorizerState, DictVectorizerState, TfidfTransformerState
from .witnesses import (
    witness_count_vectorizer_analyze,
    witness_count_vectorizer_feature_names,
    witness_count_vectorizer_fit,
    witness_count_vectorizer_inverse_transform,
    witness_count_vectorizer_transform,
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
VocabularySpec = tuple[str, ...] | dict[str, int]


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


def _raw_documents_valid(raw_documents: tuple[str, ...]) -> bool:
    return bool(isinstance(raw_documents, tuple) and len(raw_documents) > 0 and all(isinstance(doc, str) for doc in raw_documents))


def _strip_accents_valid(strip_accents: str | None) -> bool:
    return strip_accents in {None, "ascii", "unicode"}


def _ngram_range_valid(ngram_range: tuple[int, int]) -> bool:
    return bool(
        isinstance(ngram_range, tuple)
        and len(ngram_range) == 2
        and all(isinstance(value, int) and not isinstance(value, bool) for value in ngram_range)
        and ngram_range[0] >= 1
        and ngram_range[0] <= ngram_range[1]
    )


def _token_pattern_valid(token_pattern: str) -> bool:
    if not isinstance(token_pattern, str) or not token_pattern:
        return False
    try:
        pattern = re.compile(token_pattern)
    except re.error:
        return False
    return bool(pattern.groups <= 1)


def _stop_words_valid(stop_words: tuple[str, ...] | None) -> bool:
    return bool(stop_words is None or (isinstance(stop_words, tuple) and all(isinstance(word, str) for word in stop_words)))


def _df_threshold_valid(value: int | float) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return value >= 1
    if isinstance(value, float):
        return 0.0 <= value <= 1.0
    return False


def _max_features_valid(max_features: int | None) -> bool:
    return bool(max_features is None or (isinstance(max_features, int) and not isinstance(max_features, bool) and max_features >= 1))


def _vocabulary_valid(vocabulary: VocabularySpec | None) -> bool:
    if vocabulary is None:
        return True
    if isinstance(vocabulary, tuple):
        return bool(len(vocabulary) > 0 and len(set(vocabulary)) == len(vocabulary) and all(isinstance(term, str) and term for term in vocabulary))
    if isinstance(vocabulary, dict):
        indices = list(vocabulary.values())
        return bool(
            len(vocabulary) > 0
            and all(isinstance(term, str) and term for term in vocabulary)
            and all(isinstance(index, int) and not isinstance(index, bool) for index in indices)
            and sorted(indices) == list(range(len(indices)))
        )
    return False


def _count_vectorizer_config_valid(
    raw_documents: tuple[str, ...],
    strip_accents: str | None,
    token_pattern: str,
    ngram_range: tuple[int, int],
    stop_words: tuple[str, ...] | None,
    max_df: int | float,
    min_df: int | float,
    max_features: int | None,
    vocabulary: VocabularySpec | None,
) -> bool:
    return bool(
        _raw_documents_valid(raw_documents)
        and _strip_accents_valid(strip_accents)
        and _token_pattern_valid(token_pattern)
        and _ngram_range_valid(ngram_range)
        and _stop_words_valid(stop_words)
        and _df_threshold_valid(max_df)
        and _df_threshold_valid(min_df)
        and _max_features_valid(max_features)
        and _vocabulary_valid(vocabulary)
    )


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


def _count_vectorizer_state_valid(state: CountVectorizerState) -> bool:
    return bool(
        len(state.feature_names) >= 1
        and len(state.feature_names) == len(state.vocabulary)
        and set(state.feature_names) == set(state.vocabulary)
        and sorted(state.vocabulary.values()) == list(range(len(state.feature_names)))
        and all(state.feature_names[index] == term for term, index in state.vocabulary.items())
        and _strip_accents_valid(state.strip_accents)
        and _token_pattern_valid(state.token_pattern)
        and _ngram_range_valid(state.ngram_range)
        and _stop_words_valid(state.stop_words)
        and isinstance(state.lowercase, bool)
        and isinstance(state.binary, bool)
        and isinstance(state.fixed_vocabulary, bool)
    )


def _analyze_result_valid(result: tuple[str, ...]) -> bool:
    return bool(isinstance(result, tuple) and all(isinstance(token, str) and token for token in result))


def _count_fit_result_valid(result: CountVectorizerState) -> bool:
    return _count_vectorizer_state_valid(result)


def _count_transform_result_valid(result: NDArray[np.float64], raw_documents: tuple[str, ...], state: CountVectorizerState) -> bool:
    values = np.asarray(result, dtype=np.float64)
    return bool(
        values.shape == (len(raw_documents), len(state.feature_names))
        and np.all(np.isfinite(values))
        and np.all(values >= 0.0)
        and (not state.binary or np.all((values == 0.0) | (values == 1.0)))
    )


def _count_inverse_result_valid(result: list[tuple[str, ...]], X: NDArray[np.float64]) -> bool:
    return bool(len(result) == np.asarray(X).shape[0] and all(isinstance(row, tuple) and all(isinstance(term, str) for term in row) for row in result))


def _feature_name(key: str, value: FeatureValue, separator: str) -> str | None:
    if isinstance(value, str):
        return f"{key}{separator}{value}"
    if isinstance(value, (int, float)):
        return key
    return None


def _iterable_feature_names(key: str, values: tuple[str, ...], separator: str) -> tuple[str, ...]:
    return tuple(f"{key}{separator}{value}" for value in values)


def _strip_accents_unicode(text: str) -> str:
    try:
        text.encode("ASCII", errors="strict")
        return text
    except UnicodeEncodeError:
        normalized = unicodedata.normalize("NFKD", text)
        return "".join([char for char in normalized if not unicodedata.combining(char)])


def _strip_accents_ascii(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return normalized.encode("ASCII", "ignore").decode("ASCII")


def _preprocess_text(document: str, *, lowercase: bool, strip_accents: str | None) -> str:
    text = document.lower() if lowercase else document
    if strip_accents == "ascii":
        return _strip_accents_ascii(text)
    if strip_accents == "unicode":
        return _strip_accents_unicode(text)
    return text


def _word_ngrams(tokens: list[str], ngram_range: tuple[int, int], stop_words: tuple[str, ...] | None) -> tuple[str, ...]:
    if stop_words is not None:
        stop_set = set(stop_words)
        tokens = [token for token in tokens if token not in stop_set]
    min_n, max_n = ngram_range
    if max_n == 1:
        return tuple(tokens)
    original_tokens = tokens
    output = list(original_tokens) if min_n == 1 else []
    start_n = 2 if min_n == 1 else min_n
    n_original_tokens = len(original_tokens)
    for n in range(start_n, min(max_n + 1, n_original_tokens + 1)):
        for i in range(n_original_tokens - n + 1):
            output.append(" ".join(original_tokens[i : i + n]))
    return tuple(output)


def _normalize_vocabulary(vocabulary: VocabularySpec) -> dict[str, int]:
    if isinstance(vocabulary, dict):
        return dict(vocabulary)
    return {term: index for index, term in enumerate(vocabulary)}


def _matrix_from_documents(
    raw_documents: tuple[str, ...],
    vocabulary: dict[str, int],
    *,
    lowercase: bool,
    strip_accents: str | None,
    token_pattern: str,
    ngram_range: tuple[int, int],
    stop_words: tuple[str, ...] | None,
    binary: bool,
) -> NDArray[np.float64]:
    matrix = np.zeros((len(raw_documents), len(vocabulary)), dtype=np.float64)
    for row, document in enumerate(raw_documents):
        for token in count_vectorizer_analyze(
            document,
            lowercase=lowercase,
            strip_accents=strip_accents,
            token_pattern=token_pattern,
            ngram_range=ngram_range,
            stop_words=stop_words,
        ):
            if token in vocabulary:
                matrix[row, vocabulary[token]] += 1.0
    if binary:
        matrix[matrix > 0.0] = 1.0
    return matrix


def _sort_count_features(matrix: NDArray[np.float64], vocabulary: dict[str, int]) -> tuple[NDArray[np.float64], dict[str, int]]:
    sorted_features = sorted(vocabulary.items())
    old_indices = [old_index for _term, old_index in sorted_features]
    sorted_matrix = matrix[:, old_indices]
    sorted_vocabulary = {term: new_index for new_index, (term, _old_index) in enumerate(sorted_features)}
    return sorted_matrix, sorted_vocabulary


def _limit_count_features(
    matrix: NDArray[np.float64],
    vocabulary: dict[str, int],
    *,
    high: float | None,
    low: float | None,
    limit: int | None,
) -> tuple[NDArray[np.float64], dict[str, int]]:
    if high is None and low is None and limit is None:
        return matrix, vocabulary
    document_frequency = tfidf_document_frequency(matrix)
    mask = np.ones(document_frequency.shape[0], dtype=bool)
    if high is not None:
        mask &= document_frequency <= high
    if low is not None:
        mask &= document_frequency >= low
    if limit is not None and int(mask.sum()) > limit:
        term_frequency = np.asarray(np.sum(matrix, axis=0), dtype=np.float64)
        selected_relative = (-term_frequency[mask]).argsort(kind="quicksort")[:limit]
        new_mask = np.zeros(mask.shape, dtype=bool)
        new_mask[np.where(mask)[0][selected_relative]] = True
        mask = new_mask
    kept_indices = np.where(mask)[0]
    if kept_indices.size == 0:
        raise ValueError("After pruning, no terms remain. Try a lower min_df or a higher max_df.")
    new_vocabulary: dict[str, int] = {}
    for term, old_index in vocabulary.items():
        if mask[old_index]:
            new_vocabulary[term] = int(np.searchsorted(kept_indices, old_index))
    return matrix[:, kept_indices], new_vocabulary


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


@register_atom(witness_count_vectorizer_analyze)
@icontract.require(lambda document: isinstance(document, str), "document must be text")
@icontract.require(lambda strip_accents: _strip_accents_valid(strip_accents), "strip_accents must be None, 'ascii', or 'unicode'")
@icontract.require(lambda token_pattern: _token_pattern_valid(token_pattern), "token_pattern must compile with at most one capture group")
@icontract.require(lambda ngram_range: _ngram_range_valid(ngram_range), "ngram_range must be a valid positive integer interval")
@icontract.require(lambda stop_words: _stop_words_valid(stop_words), "stop_words must be None or a tuple of strings")
@icontract.ensure(lambda result: _analyze_result_valid(result), "analyzed tokens must be strings")
def count_vectorizer_analyze(
    document: str,
    *,
    lowercase: bool = True,
    strip_accents: str | None = None,
    token_pattern: str = r"(?u)\b\w\w+\b",
    ngram_range: tuple[int, int] = (1, 1),
    stop_words: tuple[str, ...] | None = None,
) -> tuple[str, ...]:
    """Analyze one text document into word n-gram tokens."""
    text = _preprocess_text(document, lowercase=bool(lowercase), strip_accents=strip_accents)
    tokens = re.compile(token_pattern).findall(text)
    if tokens and isinstance(tokens[0], tuple):
        tokens = [match[0] for match in tokens]
    return _word_ngrams(list(tokens), ngram_range, stop_words)


@register_atom(witness_count_vectorizer_fit)
@icontract.require(
    lambda raw_documents, strip_accents, token_pattern, ngram_range, stop_words, max_df, min_df, max_features, vocabulary: _count_vectorizer_config_valid(
        raw_documents,
        strip_accents,
        token_pattern,
        ngram_range,
        stop_words,
        max_df,
        min_df,
        max_features,
        vocabulary,
    ),
    "raw documents and CountVectorizer configuration must be supported",
)
@icontract.ensure(lambda result: _count_fit_result_valid(result), "state must contain a valid count-vectorizer vocabulary")
def count_vectorizer_fit(
    raw_documents: tuple[str, ...],
    *,
    lowercase: bool = True,
    strip_accents: str | None = None,
    token_pattern: str = r"(?u)\b\w\w+\b",
    ngram_range: tuple[int, int] = (1, 1),
    stop_words: tuple[str, ...] | None = None,
    max_df: int | float = 1.0,
    min_df: int | float = 1,
    max_features: int | None = None,
    vocabulary: VocabularySpec | None = None,
    binary: bool = False,
) -> CountVectorizerState:
    """Fit a dense word-count vectorizer vocabulary from text documents."""
    fixed_vocabulary = vocabulary is not None
    if fixed_vocabulary:
        learned_vocabulary = _normalize_vocabulary(vocabulary)
    else:
        learned_vocabulary: dict[str, int] = {}
        for document in raw_documents:
            for token in count_vectorizer_analyze(
                document,
                lowercase=lowercase,
                strip_accents=strip_accents,
                token_pattern=token_pattern,
                ngram_range=ngram_range,
                stop_words=stop_words,
            ):
                if token not in learned_vocabulary:
                    learned_vocabulary[token] = len(learned_vocabulary)
        if not learned_vocabulary:
            raise ValueError("empty vocabulary; perhaps the documents only contain stop words")

        matrix = _matrix_from_documents(
            raw_documents,
            learned_vocabulary,
            lowercase=lowercase,
            strip_accents=strip_accents,
            token_pattern=token_pattern,
            ngram_range=ngram_range,
            stop_words=stop_words,
            binary=bool(binary),
        )
        n_doc = len(raw_documents)
        max_doc_count = float(max_df) if isinstance(max_df, int) else float(max_df) * n_doc
        min_doc_count = float(min_df) if isinstance(min_df, int) else float(min_df) * n_doc
        if max_doc_count < min_doc_count:
            raise ValueError("max_df corresponds to fewer documents than min_df")
        if max_features is not None:
            matrix, learned_vocabulary = _sort_count_features(matrix, learned_vocabulary)
        matrix, learned_vocabulary = _limit_count_features(
            matrix,
            learned_vocabulary,
            high=max_doc_count,
            low=min_doc_count,
            limit=max_features,
        )
        if max_features is None:
            matrix, learned_vocabulary = _sort_count_features(matrix, learned_vocabulary)

    feature_names = tuple(term for term, _index in sorted(learned_vocabulary.items(), key=lambda item: item[1]))
    return CountVectorizerState(
        vocabulary=learned_vocabulary,
        feature_names=feature_names,
        lowercase=bool(lowercase),
        strip_accents=strip_accents,
        token_pattern=token_pattern,
        ngram_range=ngram_range,
        stop_words=stop_words,
        binary=bool(binary),
        fixed_vocabulary=fixed_vocabulary,
    )


@register_atom(witness_count_vectorizer_transform)
@icontract.require(lambda raw_documents: _raw_documents_valid(raw_documents), "raw_documents must be a non-empty tuple of strings")
@icontract.require(lambda state: _count_vectorizer_state_valid(state), "state must contain a valid count-vectorizer vocabulary")
@icontract.ensure(lambda result, raw_documents, state: _count_transform_result_valid(result, raw_documents, state), "count matrix must match documents and vocabulary")
def count_vectorizer_transform(raw_documents: tuple[str, ...], state: CountVectorizerState) -> NDArray[np.float64]:
    """Transform text documents into a dense token-count matrix."""
    return _matrix_from_documents(
        raw_documents,
        state.vocabulary,
        lowercase=state.lowercase,
        strip_accents=state.strip_accents,
        token_pattern=state.token_pattern,
        ngram_range=state.ngram_range,
        stop_words=state.stop_words,
        binary=state.binary,
    )


@register_atom(witness_count_vectorizer_feature_names)
@icontract.require(lambda state: _count_vectorizer_state_valid(state), "state must contain a valid count-vectorizer vocabulary")
@icontract.ensure(lambda result, state: len(result) == len(state.feature_names), "feature names must match fitted state")
def count_vectorizer_feature_names(state: CountVectorizerState) -> tuple[str, ...]:
    """Return count-vectorizer feature names in output-column order."""
    return state.feature_names


@register_atom(witness_count_vectorizer_inverse_transform)
@icontract.require(lambda X: _matrix_2d(X), "X must be a dense numeric 2D matrix")
@icontract.require(lambda state: _count_vectorizer_state_valid(state), "state must contain a valid count-vectorizer vocabulary")
@icontract.require(lambda X, state: np.asarray(X).shape[1] == len(state.feature_names), "X feature count must match fitted state")
@icontract.ensure(lambda result, X: _count_inverse_result_valid(result, X), "inverse terms must match sample count")
def count_vectorizer_inverse_transform(X: NDArray[np.float64], state: CountVectorizerState) -> list[tuple[str, ...]]:
    """Return feature terms with nonzero counts for each document row."""
    values = np.asarray(X, dtype=np.float64)
    rows: list[tuple[str, ...]] = []
    for row in range(values.shape[0]):
        rows.append(tuple(state.feature_names[index] for index in np.flatnonzero(values[row, :])))
    return rows
