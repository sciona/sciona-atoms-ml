"""Ghost witnesses for partial_dependence input bookkeeping atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_partial_dependence_feature_name_index(
    feature_name: str,
    feature_names: tuple[str, ...],
) -> AbstractArray:
    """Describe the integer index of a feature name in a provided name tuple."""
    del feature_name
    if len(feature_names) < 1:
        raise ValueError("feature_names must be nonempty")
    return AbstractArray(shape=(), dtype="int64", min_val=0.0, max_val=float(len(feature_names) - 1))


def witness_partial_dependence_nonnegative_int_features(
    features: AbstractArray,
    *,
    n_features: int,
) -> AbstractArray:
    """Describe a validated nonnegative integer feature vector."""
    del n_features
    if len(features.shape) != 1 or int(features.shape[0]) < 1:
        raise ValueError("features must be a nonempty 1D vector")
    return AbstractArray(shape=(int(features.shape[0]),), dtype="int64")


def witness_partial_dependence_default_categorical_mask(
    n_selected_features: int,
) -> AbstractArray:
    """Describe the default all-false categorical mask."""
    if n_selected_features < 1:
        raise ValueError("n_selected_features must be positive")
    return AbstractArray(shape=(n_selected_features,), dtype="bool")


def witness_partial_dependence_boolean_categorical_mask(
    categorical_features: AbstractArray,
    features_indices: AbstractArray,
    *,
    n_features: int,
) -> AbstractArray:
    """Describe a selected-feature categorical mask taken from a global boolean vector."""
    del categorical_features, n_features
    if len(features_indices.shape) != 1 or int(features_indices.shape[0]) < 1:
        raise ValueError("features_indices must be nonempty")
    return AbstractArray(shape=(int(features_indices.shape[0]),), dtype="bool")


def witness_partial_dependence_index_categorical_mask(
    categorical_indices: AbstractArray,
    features_indices: AbstractArray,
) -> AbstractArray:
    """Describe a selected-feature categorical mask built from categorical indices."""
    del categorical_indices
    if len(features_indices.shape) != 1 or int(features_indices.shape[0]) < 1:
        raise ValueError("features_indices must be nonempty")
    return AbstractArray(shape=(int(features_indices.shape[0]),), dtype="bool")


def witness_partial_dependence_name_categorical_mask(
    categorical_names: tuple[str, ...],
    features_indices: AbstractArray,
    feature_names: tuple[str, ...],
) -> AbstractArray:
    """Describe a selected-feature categorical mask built from categorical feature names."""
    del categorical_names, feature_names
    if len(features_indices.shape) != 1 or int(features_indices.shape[0]) < 1:
        raise ValueError("features_indices must be nonempty")
    return AbstractArray(shape=(int(features_indices.shape[0]),), dtype="bool")
