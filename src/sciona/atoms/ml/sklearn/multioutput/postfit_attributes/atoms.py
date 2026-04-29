"""Multioutput post-fit attribute helpers adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    ClassVectorTuple,
    witness_multioutput_classifier_classes,
    witness_multioutput_fit_feature_names_in,
    witness_multioutput_fit_n_features_in,
    witness_multioutput_partial_fit_feature_names_in_update_required,
    witness_multioutput_partial_fit_n_features_in_update_required,
)


def _flag_valid(value: object) -> bool:
    return isinstance(value, bool)


def _positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


def _feature_names_valid(feature_names: object) -> bool:
    return bool(
        isinstance(feature_names, tuple)
        and len(feature_names) >= 1
        and all(isinstance(name, str) and name != "" for name in feature_names)
    )


def _class_vector_valid(values: object) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(
        array.ndim == 1
        and array.shape[0] >= 1
        and np.all(np.isfinite(array))
        and np.unique(array).shape[0] == array.shape[0]
    )


def _class_vector_tuple_valid(values: object) -> bool:
    return bool(
        isinstance(values, tuple)
        and len(values) >= 1
        and all(_class_vector_valid(item) for item in values)
    )


def _class_vector_tuple_result_valid(result: object, class_vectors: object) -> bool:
    if not isinstance(result, tuple) or not isinstance(class_vectors, tuple) or len(result) != len(class_vectors):
        return False
    return all(
        _class_vector_valid(observed)
        and np.array_equal(np.asarray(observed, dtype=np.float64), np.asarray(expected, dtype=np.float64))
        for observed, expected in zip(result, class_vectors)
    )


@register_atom(witness_multioutput_partial_fit_n_features_in_update_required)
@icontract.require(lambda first_time: _flag_valid(first_time), "first_time must be boolean")
@icontract.require(lambda estimator_has_n_features_in: _flag_valid(estimator_has_n_features_in), "estimator_has_n_features_in must be boolean")
@icontract.ensure(lambda result: _flag_valid(result), "result must be boolean")
def multioutput_partial_fit_n_features_in_update_required(
    *,
    first_time: bool,
    estimator_has_n_features_in: bool,
) -> bool:
    """Return whether sklearn partial_fit should expose n_features_in_ from the first estimator."""
    return bool(first_time and estimator_has_n_features_in)


@register_atom(witness_multioutput_partial_fit_feature_names_in_update_required)
@icontract.require(lambda first_time: _flag_valid(first_time), "first_time must be boolean")
@icontract.require(lambda estimator_has_feature_names_in: _flag_valid(estimator_has_feature_names_in), "estimator_has_feature_names_in must be boolean")
@icontract.ensure(lambda result: _flag_valid(result), "result must be boolean")
def multioutput_partial_fit_feature_names_in_update_required(
    *,
    first_time: bool,
    estimator_has_feature_names_in: bool,
) -> bool:
    """Return whether sklearn partial_fit should expose feature_names_in_ from the first estimator."""
    return bool(first_time and estimator_has_feature_names_in)


@register_atom(witness_multioutput_fit_n_features_in)
@icontract.require(lambda n_features_in: _positive_int(n_features_in), "n_features_in must be a positive integer")
@icontract.ensure(lambda result: _positive_int(result), "result must be a positive integer")
def multioutput_fit_n_features_in(n_features_in: int) -> int:
    """Expose sklearn's fitted n_features_in_ value from the first trained estimator."""
    return int(n_features_in)


@register_atom(witness_multioutput_fit_feature_names_in)
@icontract.require(lambda feature_names_in: _feature_names_valid(feature_names_in), "feature_names_in must be a nonempty tuple of nonempty strings")
@icontract.ensure(lambda result: _feature_names_valid(result), "result must preserve a nonempty tuple of feature names")
def multioutput_fit_feature_names_in(feature_names_in: tuple[str, ...]) -> tuple[str, ...]:
    """Expose sklearn's fitted feature_names_in_ tuple from the first trained estimator."""
    return tuple(feature_names_in)


@register_atom(witness_multioutput_classifier_classes)
@icontract.require(lambda class_vectors: _class_vector_tuple_valid(class_vectors), "class_vectors must be a nonempty tuple of finite unique class vectors")
@icontract.ensure(lambda result, class_vectors: _class_vector_tuple_result_valid(result, class_vectors), "result must preserve one class vector per fitted estimator")
def multioutput_classifier_classes(
    class_vectors: ClassVectorTuple,
) -> ClassVectorTuple:
    """Collect sklearn's per-estimator classes_ arrays for a fitted MultiOutputClassifier or ClassifierChain."""
    return tuple(np.asarray(class_vector, dtype=np.float64) for class_vector in class_vectors)
