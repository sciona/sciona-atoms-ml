"""Witnesses for DictionaryLearning output/tag helpers."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_dictionary_learning_fit_return_self(estimator_token: str) -> AbstractArray:
    """Describe DictionaryLearning.fit returning self unchanged."""
    del estimator_token
    return AbstractArray(shape=(), dtype="object")


def witness_dictionary_learning_n_features_out(components: AbstractArray) -> AbstractArray:
    """Describe DictionaryLearning._n_features_out."""
    return AbstractArray(shape=(), dtype="int64")


def witness_dictionary_learning_preserves_dtype_tags(parent_preserves_dtype: AbstractArray) -> AbstractArray:
    """Describe DictionaryLearning preserves_dtype tag override."""
    del parent_preserves_dtype
    return AbstractArray(shape=(2,), dtype="object")


def witness_dictionary_learning_minibatch_n_features_out(components: AbstractArray) -> AbstractArray:
    """Describe MiniBatchDictionaryLearning._n_features_out."""
    return AbstractArray(shape=(), dtype="int64")


def witness_dictionary_learning_minibatch_preserves_dtype_tags(parent_preserves_dtype: AbstractArray) -> AbstractArray:
    """Describe MiniBatchDictionaryLearning preserves_dtype tag override."""
    del parent_preserves_dtype
    return AbstractArray(shape=(2,), dtype="object")
