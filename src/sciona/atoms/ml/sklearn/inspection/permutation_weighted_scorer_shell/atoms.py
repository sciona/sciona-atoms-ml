"""Permutation weighted-scorer shell atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_permutation_importance_scorer_kwargs,
    witness_permutation_importance_use_sample_weight,
)

WeightVector = NDArray[np.float64]


def _optional_weight_vector(value: object) -> bool:
    if value is None:
        return True
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 1 and array.shape[0] >= 1 and np.all(np.isfinite(array)))


def _kwargs_valid(result: object, sample_weight: object) -> bool:
    if sample_weight is None:
        return isinstance(result, dict) and not result
    if not isinstance(result, dict) or set(result) != {"sample_weight"}:
        return False
    expected = np.asarray(sample_weight, dtype=np.float64)
    actual = np.asarray(result["sample_weight"], dtype=np.float64)
    return bool(actual.shape == expected.shape and np.array_equal(actual, expected))


@register_atom(witness_permutation_importance_use_sample_weight)
@icontract.require(
    lambda sample_weight: _optional_weight_vector(sample_weight),
    "sample_weight must be None or a finite nonempty one-dimensional weight vector",
)
@icontract.ensure(lambda result: isinstance(result, bool), "result must be boolean")
def permutation_importance_use_sample_weight(
    sample_weight: WeightVector | list[float] | None,
) -> bool:
    """Decide whether permutation importance forwards sample_weight into the scorer."""
    return sample_weight is not None


@register_atom(witness_permutation_importance_scorer_kwargs)
@icontract.require(
    lambda sample_weight: _optional_weight_vector(sample_weight),
    "sample_weight must be None or a finite nonempty one-dimensional weight vector",
)
@icontract.ensure(
    lambda result, sample_weight: _kwargs_valid(result, sample_weight),
    "result must match sklearn's scorer kwargs for the sample-weight branch",
)
def permutation_importance_scorer_kwargs(
    sample_weight: WeightVector | list[float] | None,
) -> dict[str, WeightVector]:
    """Resolve the scorer kwargs used by sklearn's private _weights_scorer helper."""
    if sample_weight is None:
        return {}
    return {"sample_weight": np.asarray(sample_weight, dtype=np.float64).copy()}
