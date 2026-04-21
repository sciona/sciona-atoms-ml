"""Ghost witnesses for sklearn feature extraction atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray

from .state_models import DictVectorizerState, TfidfTransformerState


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
