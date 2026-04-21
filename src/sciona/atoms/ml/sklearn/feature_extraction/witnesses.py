"""Ghost witnesses for sklearn feature extraction atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray

from .state_models import DictVectorizerState


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
