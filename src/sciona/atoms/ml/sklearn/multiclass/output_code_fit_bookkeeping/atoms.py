"""Output-code fit bookkeeping helpers adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_output_code_fit_estimator_count,
    witness_output_code_fit_feature_names_in,
    witness_output_code_fit_n_features_in,
    witness_output_code_fit_require_nonempty_classes,
)


def _class_vector_valid(values: object, *, min_classes: int = 0) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(
        array.ndim == 1
        and array.shape[0] >= min_classes
        and np.all(np.isfinite(array))
        and np.unique(array).shape[0] == array.shape[0]
    )


def _finite_positive_float(value: object) -> bool:
    return bool(isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(float(value)) and float(value) > 0.0)


def _positive_int(value: object) -> bool:
    return bool(isinstance(value, int) and not isinstance(value, bool) and value >= 1)


def _feature_names_valid(feature_names: object) -> bool:
    return bool(
        isinstance(feature_names, tuple)
        and len(feature_names) >= 1
        and all(isinstance(name, str) and name != "" for name in feature_names)
    )


@register_atom(witness_output_code_fit_require_nonempty_classes)
@icontract.require(lambda classes: _class_vector_valid(classes, min_classes=0), "classes must be a finite unique class vector")
@icontract.ensure(lambda result: _class_vector_valid(result, min_classes=1), "validated classes must contain at least one unique class value")
def output_code_fit_require_nonempty_classes(
    classes: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Require sklearn's output-code fit guard that rejects an empty class vector."""
    class_values = np.asarray(classes, dtype=np.float64)
    if class_values.shape[0] == 0:
        raise ValueError("OutputCodeClassifier can not be fit when no class is present.")
    return np.asarray(class_values, dtype=np.float64)


@register_atom(witness_output_code_fit_estimator_count)
@icontract.require(lambda n_classes: _positive_int(n_classes), "n_classes must be a positive integer")
@icontract.require(lambda code_size: _finite_positive_float(code_size), "code_size must be a finite positive float")
@icontract.ensure(lambda result: _positive_int(result), "estimator count must be a positive integer")
def output_code_fit_estimator_count(
    n_classes: int,
    code_size: float,
) -> int:
    """Resolve sklearn's output-code estimator count from class count and code_size."""
    return int(int(n_classes) * float(code_size))


@register_atom(witness_output_code_fit_n_features_in)
@icontract.require(lambda n_features_in: _positive_int(n_features_in), "n_features_in must be a positive integer")
@icontract.ensure(lambda result: _positive_int(result), "result must be a positive integer")
def output_code_fit_n_features_in(n_features_in: int) -> int:
    """Expose sklearn's fitted n_features_in_ copied from the first trained output-code estimator."""
    return int(n_features_in)


@register_atom(witness_output_code_fit_feature_names_in)
@icontract.require(lambda feature_names_in: _feature_names_valid(feature_names_in), "feature_names_in must be a nonempty tuple of nonempty strings")
@icontract.ensure(lambda result: _feature_names_valid(result), "result must preserve a nonempty tuple of feature names")
def output_code_fit_feature_names_in(feature_names_in: tuple[str, ...]) -> tuple[str, ...]:
    """Expose sklearn's fitted feature_names_in_ tuple copied from the first trained output-code estimator."""
    return tuple(feature_names_in)
