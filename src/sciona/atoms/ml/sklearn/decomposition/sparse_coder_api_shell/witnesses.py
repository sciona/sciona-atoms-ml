"""Witnesses for sklearn SparseCoder API-shell helpers."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_sparse_coder_fit_return_self(estimator_token: str) -> AbstractArray:
    """Describe SparseCoder.fit returning self unchanged."""
    del estimator_token
    return AbstractArray(shape=(), dtype="object")


def witness_sparse_coder_transform_dictionary(dictionary: AbstractArray) -> AbstractArray:
    """Describe the dictionary passed into _transform by SparseCoder.transform."""
    return AbstractArray(shape=dictionary.shape, dtype=dictionary.dtype)


def witness_sparse_coder_requires_fit_tag(parent_requires_fit: bool) -> AbstractArray:
    """Describe the requires_fit tag override."""
    del parent_requires_fit
    return AbstractArray(shape=(), dtype="bool")


def witness_sparse_coder_preserves_dtype_tags(parent_preserves_dtype: AbstractArray) -> AbstractArray:
    """Describe the preserves_dtype tag override."""
    del parent_preserves_dtype
    return AbstractArray(shape=(2,), dtype="object")


def witness_sparse_coder_n_features_out(dictionary: AbstractArray) -> AbstractArray:
    """Describe SparseCoder._n_features_out exposure."""
    return AbstractArray(shape=(), dtype="int64")
