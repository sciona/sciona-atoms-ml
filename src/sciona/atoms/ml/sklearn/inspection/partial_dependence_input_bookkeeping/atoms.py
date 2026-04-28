"""partial_dependence input bookkeeping helpers adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_partial_dependence_boolean_categorical_mask,
    witness_partial_dependence_default_categorical_mask,
    witness_partial_dependence_feature_name_index,
    witness_partial_dependence_index_categorical_mask,
    witness_partial_dependence_name_categorical_mask,
    witness_partial_dependence_nonnegative_int_features,
)


def _feature_count_valid(n_features: int) -> bool:
    return bool(isinstance(n_features, int) and not isinstance(n_features, bool) and n_features >= 1)


def _feature_indices_valid(features_indices: NDArray[np.int64]) -> bool:
    values = np.asarray(features_indices)
    return bool(values.ndim == 1 and values.shape[0] >= 1 and np.issubdtype(values.dtype, np.integer))


def _feature_name_tuple_valid(feature_names: tuple[str, ...]) -> bool:
    return bool(
        len(feature_names) >= 1
        and all(isinstance(name, str) and len(name) >= 1 for name in feature_names)
        and len(set(feature_names)) == len(feature_names)
    )


def _categorical_bool_mask_valid(
    result: tuple[bool, ...],
    features_indices: NDArray[np.int64],
) -> bool:
    values = np.asarray(features_indices)
    return bool(
        isinstance(result, tuple)
        and len(result) == values.shape[0]
        and all(isinstance(item, bool) for item in result)
    )


def _categorical_index_vector_valid(categorical_indices: NDArray[np.int64]) -> bool:
    values = np.asarray(categorical_indices)
    return bool(values.ndim == 1 and np.issubdtype(values.dtype, np.integer))


@register_atom(witness_partial_dependence_feature_name_index)
@icontract.require(lambda feature_name: isinstance(feature_name, str) and len(feature_name) >= 1, "feature_name must be a nonempty string")
@icontract.require(lambda feature_names: _feature_name_tuple_valid(feature_names), "feature_names must be a nonempty tuple of unique strings")
@icontract.require(lambda feature_name, feature_names: feature_name in feature_names, "feature_name must exist in feature_names")
@icontract.ensure(lambda result, feature_names: isinstance(result, int) and 0 <= result < len(feature_names), "result must be a valid feature_names index")
def partial_dependence_feature_name_index(
    feature_name: str,
    feature_names: tuple[str, ...],
) -> int:
    """Resolve sklearn's feature-name lookup used by partial_dependence."""
    return int(feature_names.index(feature_name))


@register_atom(witness_partial_dependence_nonnegative_int_features)
@icontract.require(lambda features: _feature_indices_valid(features), "features must be a nonempty integer vector")
@icontract.require(lambda n_features: _feature_count_valid(n_features), "n_features must be a positive integer")
@icontract.require(lambda features: np.all(np.asarray(features, dtype=np.int64) >= 0), "integer features must be nonnegative")
@icontract.ensure(lambda result, features: np.array_equal(np.asarray(result, dtype=np.int64), np.asarray(features, dtype=np.int64)), "validated features must preserve the input vector")
def partial_dependence_nonnegative_int_features(
    features: NDArray[np.int64],
    *,
    n_features: int,
) -> NDArray[np.int64]:
    """Validate sklearn's nonnegative integer feature requirement before column lookup."""
    del n_features
    return np.asarray(features, dtype=np.int64)


@register_atom(witness_partial_dependence_default_categorical_mask)
@icontract.require(lambda n_selected_features: _feature_count_valid(n_selected_features), "n_selected_features must be a positive integer")
@icontract.ensure(lambda result, n_selected_features: isinstance(result, tuple) and result == tuple(False for _ in range(n_selected_features)), "default categorical mask must be all False")
def partial_dependence_default_categorical_mask(
    n_selected_features: int,
) -> tuple[bool, ...]:
    """Return sklearn's default all-false categorical mask for selected features."""
    return tuple(False for _ in range(n_selected_features))


@register_atom(witness_partial_dependence_boolean_categorical_mask)
@icontract.require(lambda categorical_features: np.asarray(categorical_features).ndim == 1 and np.asarray(categorical_features).dtype == np.bool_, "categorical_features must be a 1D boolean mask")
@icontract.require(lambda features_indices: _feature_indices_valid(features_indices), "features_indices must be a nonempty integer vector")
@icontract.require(lambda n_features: _feature_count_valid(n_features), "n_features must be a positive integer")
@icontract.require(lambda categorical_features, n_features: np.asarray(categorical_features).shape[0] == n_features, "categorical_features mask length must match n_features")
@icontract.ensure(lambda result, features_indices: _categorical_bool_mask_valid(result, features_indices), "result must be a boolean tuple aligned with features_indices")
def partial_dependence_boolean_categorical_mask(
    categorical_features: NDArray[np.bool_],
    features_indices: NDArray[np.int64],
    *,
    n_features: int,
) -> tuple[bool, ...]:
    """Map sklearn's global boolean categorical mask onto the selected feature indices."""
    del n_features
    mask = np.asarray(categorical_features, dtype=np.bool_)
    selected = np.asarray(features_indices, dtype=np.int64)
    return tuple(bool(mask[idx]) for idx in selected)


@register_atom(witness_partial_dependence_index_categorical_mask)
@icontract.require(lambda categorical_indices: _categorical_index_vector_valid(categorical_indices), "categorical_indices must be an integer vector")
@icontract.require(lambda features_indices: _feature_indices_valid(features_indices), "features_indices must be a nonempty integer vector")
@icontract.ensure(lambda result, features_indices: _categorical_bool_mask_valid(result, features_indices), "result must be a boolean tuple aligned with features_indices")
def partial_dependence_index_categorical_mask(
    categorical_indices: NDArray[np.int64],
    features_indices: NDArray[np.int64],
) -> tuple[bool, ...]:
    """Resolve sklearn's categorical-membership mask from categorical feature indices."""
    categorical = set(np.asarray(categorical_indices, dtype=np.int64).tolist())
    selected = np.asarray(features_indices, dtype=np.int64)
    return tuple(int(idx) in categorical for idx in selected)


@register_atom(witness_partial_dependence_name_categorical_mask)
@icontract.require(lambda categorical_names: _feature_name_tuple_valid(categorical_names), "categorical_names must be a nonempty tuple of unique strings")
@icontract.require(lambda feature_names: _feature_name_tuple_valid(feature_names), "feature_names must be a nonempty tuple of unique strings")
@icontract.require(lambda categorical_names, feature_names: set(categorical_names).issubset(set(feature_names)), "categorical_names must be drawn from feature_names")
@icontract.require(lambda features_indices: _feature_indices_valid(features_indices), "features_indices must be a nonempty integer vector")
@icontract.ensure(lambda result, features_indices: _categorical_bool_mask_valid(result, features_indices), "result must be a boolean tuple aligned with features_indices")
def partial_dependence_name_categorical_mask(
    categorical_names: tuple[str, ...],
    features_indices: NDArray[np.int64],
    feature_names: tuple[str, ...],
) -> tuple[bool, ...]:
    """Resolve sklearn's categorical-membership mask from categorical feature names."""
    categorical_indices = {
        partial_dependence_feature_name_index(name, feature_names)
        for name in categorical_names
    }
    selected = np.asarray(features_indices, dtype=np.int64)
    return tuple(int(idx) in categorical_indices for idx in selected)
