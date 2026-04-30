"""RFE post-fit attribute helpers adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_rfe_classes,
    witness_rfe_estimator_type,
    witness_rfe_support_mask,
)


def _estimator_type_valid(value: object) -> bool:
    return bool(isinstance(value, str) and value != "")


def _classes_valid(value: object) -> bool:
    return bool(isinstance(value, tuple) and len(value) >= 1)


def _support_mask_valid(value: object) -> bool:
    array = np.asarray(value)
    return bool(array.ndim == 1 and array.shape[0] >= 1 and array.dtype == np.bool_)


def _same_bool_vector(result: object, source: object) -> bool:
    return bool(np.array_equal(np.asarray(result, dtype=np.bool_), np.asarray(source, dtype=np.bool_)))


@register_atom(witness_rfe_estimator_type)
@icontract.require(lambda estimator_type: _estimator_type_valid(estimator_type), "estimator_type must be a nonempty string")
@icontract.ensure(lambda result, estimator_type: result == estimator_type, "result must preserve the estimator type label")
def rfe_estimator_type(estimator_type: str) -> str:
    """Expose sklearn RFE's delegated estimator type label."""
    return estimator_type


@register_atom(witness_rfe_classes)
@icontract.require(lambda classes: _classes_valid(classes), "classes must be a nonempty tuple")
@icontract.ensure(lambda result, classes: result == classes, "result must preserve the fitted class labels")
def rfe_classes(classes: tuple[object, ...]) -> tuple[object, ...]:
    """Expose sklearn RFE's fitted class-label tuple from the final estimator."""
    return tuple(classes)


@register_atom(witness_rfe_support_mask)
@icontract.require(lambda support_mask: _support_mask_valid(support_mask), "support_mask must be a nonempty boolean vector")
@icontract.ensure(lambda result, support_mask: _same_bool_vector(result, support_mask), "result must preserve the fitted support mask")
def rfe_support_mask(support_mask: NDArray[np.bool_]) -> NDArray[np.bool_]:
    """Expose sklearn RFE's fitted support mask through the _get_support_mask interface."""
    return np.asarray(support_mask, dtype=np.bool_).copy()
