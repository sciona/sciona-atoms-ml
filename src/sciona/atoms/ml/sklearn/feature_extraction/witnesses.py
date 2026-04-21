"""Ghost witnesses for sklearn feature extraction atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray

from .state_models import CountVectorizerState, DictVectorizerState, TfidfTransformerState


def witness_dict_vectorizer_fit(records: tuple[dict[str, object], ...], *, separator: str = "=", sort: bool = True) -> AbstractArray:
    """Describe vocabulary learning from feature dictionaries."""
    if not records:
        raise ValueError("records must not be empty")
    if not separator:
        raise ValueError("separator must not be empty")
    return AbstractArray(shape=(len(records),), dtype="object")


def witness_dict_vectorizer_transform(records: tuple[dict[str, object], ...], state: DictVectorizerState) -> AbstractArray:
    """Describe dense matrix output for fitted dictionary vectorization."""
    if not records:
        raise ValueError("records must not be empty")
    return AbstractArray(shape=(len(records), len(state.feature_names)), dtype="float64")


def witness_dict_vectorizer_inverse_transform(X: AbstractArray, state: DictVectorizerState) -> AbstractArray:
    """Describe converting nonzero vector entries back to feature mappings."""
    if len(X.shape) != 2:
        raise ValueError("X must be 2D")
    if X.shape[1] != len(state.feature_names):
        raise ValueError("X feature count must match fitted state")
    return AbstractArray(shape=(int(X.shape[0]),), dtype="object")


def witness_dict_vectorizer_feature_names(state: DictVectorizerState) -> AbstractArray:
    """Describe the learned feature-name vector."""
    return AbstractArray(shape=(len(state.feature_names),), dtype="object")


def witness_dict_vectorizer_restrict(state: DictVectorizerState, support: tuple[int, ...]) -> AbstractArray:
    """Describe restricting a vocabulary state to selected feature columns."""
    for index in support:
        if index < 0 or index >= len(state.feature_names):
            raise ValueError("support indices must be valid")
    return AbstractArray(shape=(len(support),), dtype="object")


def witness_tfidf_document_frequency(X: AbstractArray) -> AbstractArray:
    """Describe one document-frequency count per term column."""
    if len(X.shape) != 2:
        raise ValueError("X must be 2D")
    return AbstractArray(shape=(int(X.shape[1]),), dtype="float64", min_val=0.0)


def witness_tfidf_idf(
    document_frequency: AbstractArray,
    n_samples: int,
    *,
    smooth_idf: bool = True,
) -> AbstractArray:
    """Describe one inverse-document-frequency weight per term column."""
    del smooth_idf
    if len(document_frequency.shape) != 1:
        raise ValueError("document_frequency must be 1D")
    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    return AbstractArray(shape=document_frequency.shape, dtype="float64", min_val=0.0)


def witness_tfidf_transformer_fit(
    X: AbstractArray,
    *,
    norm: str | None = "l2",
    use_idf: bool = True,
    smooth_idf: bool = True,
    sublinear_tf: bool = False,
) -> AbstractArray:
    """Describe fitting dense TF-IDF transformer state."""
    del norm, use_idf, smooth_idf, sublinear_tf
    if len(X.shape) != 2:
        raise ValueError("X must be 2D")
    return AbstractArray(shape=(int(X.shape[1]),), dtype="float64")


def witness_tfidf_transform(X: AbstractArray, state: TfidfTransformerState) -> AbstractArray:
    """Describe transforming counts into dense TF-IDF weights."""
    if len(X.shape) != 2:
        raise ValueError("X must be 2D")
    if X.shape[1] != state.n_features_in:
        raise ValueError("X feature count must match fitted state")
    return AbstractArray(shape=X.shape, dtype="float64")


def witness_count_vectorizer_analyze(
    document: str,
    *,
    lowercase: bool = True,
    strip_accents: str | None = None,
    token_pattern: str = r"(?u)\b\w\w+\b",
    ngram_range: tuple[int, int] = (1, 1),
    stop_words: tuple[str, ...] | None = None,
) -> AbstractArray:
    """Describe word-token analysis for one text document."""
    del document, lowercase, strip_accents, token_pattern, stop_words
    if ngram_range[0] < 1 or ngram_range[0] > ngram_range[1]:
        raise ValueError("ngram_range must be valid")
    return AbstractArray(shape=(1,), dtype="object")


def witness_count_vectorizer_fit(
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
    vocabulary: tuple[str, ...] | dict[str, int] | None = None,
    binary: bool = False,
) -> AbstractArray:
    """Describe fitting a dense count-vectorizer vocabulary."""
    del lowercase, strip_accents, token_pattern, stop_words, max_df, min_df, max_features, vocabulary, binary
    if not raw_documents:
        raise ValueError("raw_documents must not be empty")
    if ngram_range[0] < 1 or ngram_range[0] > ngram_range[1]:
        raise ValueError("ngram_range must be valid")
    return AbstractArray(shape=(len(raw_documents),), dtype="object")


def witness_count_vectorizer_transform(raw_documents: tuple[str, ...], state: CountVectorizerState) -> AbstractArray:
    """Describe transforming text documents into dense token counts."""
    if not raw_documents:
        raise ValueError("raw_documents must not be empty")
    return AbstractArray(shape=(len(raw_documents), len(state.feature_names)), dtype="float64")


def witness_count_vectorizer_feature_names(state: CountVectorizerState) -> AbstractArray:
    """Describe count-vectorizer feature names in output order."""
    return AbstractArray(shape=(len(state.feature_names),), dtype="object")


def witness_count_vectorizer_inverse_transform(X: AbstractArray, state: CountVectorizerState) -> AbstractArray:
    """Describe converting nonzero count columns back to terms."""
    if len(X.shape) != 2:
        raise ValueError("X must be 2D")
    if X.shape[1] != len(state.feature_names):
        raise ValueError("X feature count must match fitted state")
    return AbstractArray(shape=(int(X.shape[0]),), dtype="object")


def witness_hashing_vectorizer_token(
    token: str,
    *,
    n_features: int = 2**20,
    alternate_sign: bool = True,
) -> AbstractArray:
    """Describe hashing one token to a column/sign pair."""
    del token, alternate_sign
    if n_features < 1:
        raise ValueError("n_features must be positive")
    return AbstractArray(shape=(2,), dtype="object")


def witness_hashing_vectorizer_transform(
    raw_documents: tuple[str, ...],
    *,
    lowercase: bool = True,
    strip_accents: str | None = None,
    token_pattern: str = r"(?u)\b\w\w+\b",
    ngram_range: tuple[int, int] = (1, 1),
    stop_words: tuple[str, ...] | None = None,
    n_features: int = 2**20,
    binary: bool = False,
    norm: str | None = "l2",
    alternate_sign: bool = True,
) -> AbstractArray:
    """Describe dense hashed document-term matrix output."""
    del lowercase, strip_accents, token_pattern, stop_words, binary, norm, alternate_sign
    if not raw_documents:
        raise ValueError("raw_documents must not be empty")
    if ngram_range[0] < 1 or ngram_range[0] > ngram_range[1]:
        raise ValueError("ngram_range must be valid")
    if n_features < 1:
        raise ValueError("n_features must be positive")
    return AbstractArray(shape=(len(raw_documents), n_features), dtype="float64")
