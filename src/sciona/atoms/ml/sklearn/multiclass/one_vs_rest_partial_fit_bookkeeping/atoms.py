"""One-vs-rest partial-fit bookkeeping helpers adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_one_vs_rest_partial_fit_estimator_count,
    witness_one_vs_rest_partial_fit_label_binarizer_classes,
    witness_one_vs_rest_partial_fit_n_features_in,
)


def _positive_int(value: object) -> bool:
    return bool(isinstance(value, int) and not isinstance(value, bool) and value >= 1)


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


@register_atom(witness_one_vs_rest_partial_fit_estimator_count)
@icontract.require(lambda n_classes: _positive_int(n_classes), "n_classes must be a positive integer")
@icontract.ensure(lambda result: _positive_int(result), "estimator count must be a positive integer")
def one_vs_rest_partial_fit_estimator_count(n_classes: int) -> int:
    """Resolve sklearn's first-call OvR partial-fit estimator allocation count."""
    return int(n_classes)


@register_atom(witness_one_vs_rest_partial_fit_label_binarizer_classes)
@icontract.require(lambda classes: _class_vector_valid(classes), "classes must be a finite unique class vector")
@icontract.ensure(lambda result: _class_vector_valid(result), "label-binarizer classes must preserve a finite unique class vector")
def one_vs_rest_partial_fit_label_binarizer_classes(
    classes: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Expose the class vector used to fit sklearn's sparse OvR label binarizer on first partial_fit."""
    return np.asarray(classes, dtype=np.float64)


@register_atom(witness_one_vs_rest_partial_fit_n_features_in)
@icontract.require(lambda n_features_in: _positive_int(n_features_in), "n_features_in must be a positive integer")
@icontract.ensure(lambda result: _positive_int(result), "result must be a positive integer")
def one_vs_rest_partial_fit_n_features_in(n_features_in: int) -> int:
    """Expose sklearn's fitted n_features_in_ copied from the first OvR partial-fit estimator."""
    return int(n_features_in)
