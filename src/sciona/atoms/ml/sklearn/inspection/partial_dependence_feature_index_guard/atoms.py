"""Partial-dependence feature-index guard atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_partial_dependence_integer_feature_key_type,
    witness_partial_dependence_negative_feature_guard_required,
    witness_partial_dependence_negative_feature_message,
)


def _feature_key_type(value: object) -> bool:
    return isinstance(value, str) and len(value) >= 1


def _integer_vector(value: object) -> bool:
    try:
        array = np.asarray(value, dtype=np.int64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 1 and array.shape[0] >= 1)


def _feature_count(value: object) -> bool:
    return bool(isinstance(value, int) and not isinstance(value, bool) and value >= 1)


@register_atom(witness_partial_dependence_integer_feature_key_type)
@icontract.require(lambda key_type: _feature_key_type(key_type), "key_type must be a nonempty string")
@icontract.ensure(lambda result: isinstance(result, bool), "result must be boolean")
def partial_dependence_integer_feature_key_type(
    key_type: str,
) -> bool:
    """Decide whether partial_dependence enters the integer-feature guard branch."""
    return key_type == "int"


@register_atom(witness_partial_dependence_negative_feature_guard_required)
@icontract.require(lambda features: _integer_vector(features), "features must be a nonempty integer vector")
@icontract.ensure(lambda result: isinstance(result, bool), "result must be boolean")
def partial_dependence_negative_feature_guard_required(
    features: NDArray[np.int64],
) -> bool:
    """Decide whether partial_dependence raises for negative integer feature indices."""
    return bool(np.any(np.less(np.asarray(features, dtype=np.int64), 0)))


@register_atom(witness_partial_dependence_negative_feature_message)
@icontract.require(lambda n_features: _feature_count(n_features), "n_features must be a positive integer")
@icontract.ensure(lambda result: isinstance(result, str) and len(result) >= 1, "result must be a nonempty message")
def partial_dependence_negative_feature_message(
    n_features: int,
) -> str:
    """Format sklearn's negative-integer-feature ValueError message."""
    return "all features must be in [0, {}]".format(int(n_features) - 1)
