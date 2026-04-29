"""One-vs-rest post-fit attribute helpers adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_one_vs_rest_fit_classes,
    witness_one_vs_rest_fit_feature_names_in,
    witness_one_vs_rest_fit_n_features_in,
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


def _positive_int(value: object) -> bool:
    return bool(isinstance(value, int) and not isinstance(value, bool) and value >= 1)


def _feature_names_valid(feature_names: object) -> bool:
    return bool(
        isinstance(feature_names, tuple)
        and len(feature_names) >= 1
        and all(isinstance(name, str) and name != "" for name in feature_names)
    )


@register_atom(witness_one_vs_rest_fit_classes)
@icontract.require(lambda classes: _class_vector_valid(classes), "classes must be a finite unique class vector")
@icontract.ensure(lambda result: _class_vector_valid(result), "result must preserve a finite unique class vector")
def one_vs_rest_fit_classes(classes: NDArray[np.float64]) -> NDArray[np.float64]:
    """Expose sklearn's fitted one-vs-rest classes_ copied from the label binarizer."""
    return np.asarray(classes, dtype=np.float64)


@register_atom(witness_one_vs_rest_fit_n_features_in)
@icontract.require(lambda n_features_in: _positive_int(n_features_in), "n_features_in must be a positive integer")
@icontract.ensure(lambda result: _positive_int(result), "result must be a positive integer")
def one_vs_rest_fit_n_features_in(n_features_in: int) -> int:
    """Expose sklearn's fitted n_features_in_ copied from the first trained one-vs-rest estimator."""
    return int(n_features_in)


@register_atom(witness_one_vs_rest_fit_feature_names_in)
@icontract.require(lambda feature_names_in: _feature_names_valid(feature_names_in), "feature_names_in must be a nonempty tuple of nonempty strings")
@icontract.ensure(lambda result: _feature_names_valid(result), "result must preserve a nonempty tuple of feature names")
def one_vs_rest_fit_feature_names_in(feature_names_in: tuple[str, ...]) -> tuple[str, ...]:
    """Expose sklearn's fitted feature_names_in_ tuple copied from the first trained one-vs-rest estimator."""
    return tuple(feature_names_in)
