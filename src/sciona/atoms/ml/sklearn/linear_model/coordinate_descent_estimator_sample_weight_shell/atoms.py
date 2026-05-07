"""Sklearn coordinate-descent estimator sample-weight shell atoms."""

from __future__ import annotations

import numbers

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_cd_estimator_check_sample_weight_args,
    witness_cd_estimator_check_sample_weight_kwargs,
    witness_cd_estimator_checked_sample_weight,
    witness_cd_estimator_sample_weight_after_scalar_guard,
    witness_cd_estimator_sample_weight_check_required,
    witness_cd_estimator_sample_weight_rescale_factor,
    witness_cd_estimator_sample_weight_rescaled,
)


def _bool(value: object) -> bool:
    return isinstance(value, bool)


def _positive_int(value: object) -> bool:
    return isinstance(value, (int, np.integer)) and int(value) >= 1


def _positive_finite_sum(values: object) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim >= 1 and array.size >= 1 and np.isfinite(array).all() and np.sum(array) > 0.0)


@register_atom(witness_cd_estimator_sample_weight_after_scalar_guard)
@icontract.ensure(
    lambda result, sample_weight: (
        result is None if isinstance(sample_weight, numbers.Number) else result is sample_weight
    ),
    "scalar sample_weight must normalize to None and non-scalars must pass through",
)
def cd_estimator_sample_weight_after_scalar_guard(sample_weight: object) -> object:
    """Return sample_weight after ElasticNet.fit drops scalar numbers."""
    return None if isinstance(sample_weight, numbers.Number) else sample_weight


@register_atom(witness_cd_estimator_sample_weight_check_required)
@icontract.require(lambda check_input: _bool(check_input), "check_input must be boolean")
@icontract.ensure(
    lambda result, sample_weight, check_input: _bool(result)
    and result == (sample_weight is not None and check_input),
    "_check_sample_weight branch must match sample_weight presence and check_input",
)
def cd_estimator_sample_weight_check_required(
    sample_weight: object, check_input: bool
) -> bool:
    """Return whether ElasticNet.fit calls _check_sample_weight."""
    return sample_weight is not None and check_input


@register_atom(witness_cd_estimator_check_sample_weight_args)
@icontract.require(lambda sample_weight: sample_weight is not None, "sample_weight must be present")
@icontract.ensure(
    lambda result, sample_weight, X: isinstance(result, tuple)
    and len(result) == 2
    and result[0] is sample_weight
    and result[1] is X,
    "_check_sample_weight positional args must preserve sample_weight and X identity",
)
def cd_estimator_check_sample_weight_args(sample_weight: object, X: object) -> tuple[object, object]:
    """Return positional args for _check_sample_weight(sample_weight, X, dtype=X.dtype)."""
    return (sample_weight, X)


@register_atom(witness_cd_estimator_check_sample_weight_kwargs)
@icontract.ensure(
    lambda result, x_dtype: isinstance(result, dict) and result == {"dtype": x_dtype},
    "_check_sample_weight kwargs must map dtype through unchanged",
)
def cd_estimator_check_sample_weight_kwargs(x_dtype: object) -> dict[str, object]:
    """Return kwargs for _check_sample_weight(..., dtype=X.dtype)."""
    return {"dtype": x_dtype}


@register_atom(witness_cd_estimator_checked_sample_weight)
@icontract.ensure(
    lambda result, checked_sample_weight: result is checked_sample_weight,
    "_check_sample_weight callback result must preserve checked sample_weight identity",
)
def cd_estimator_checked_sample_weight(checked_sample_weight: object) -> object:
    """Return sample_weight after the deferred _check_sample_weight(...) callback."""
    return checked_sample_weight


@register_atom(witness_cd_estimator_sample_weight_rescale_factor)
@icontract.require(lambda sample_weight: _positive_finite_sum(sample_weight), "sample_weight must have a positive finite sum")
@icontract.require(lambda n_samples: _positive_int(n_samples), "n_samples must be positive")
@icontract.ensure(
    lambda result, sample_weight, n_samples: np.isfinite(float(result))
    and np.isclose(float(result), int(n_samples) / float(np.sum(np.asarray(sample_weight, dtype=np.float64)))),
    "rescale factor must equal n_samples / sum(sample_weight)",
)
def cd_estimator_sample_weight_rescale_factor(sample_weight: object, n_samples: int) -> float:
    """Return the factor used to rescale sample_weight to sum to n_samples."""
    return int(n_samples) / float(np.sum(np.asarray(sample_weight, dtype=np.float64)))


@register_atom(witness_cd_estimator_sample_weight_rescaled)
@icontract.require(lambda sample_weight: _positive_finite_sum(sample_weight), "sample_weight must have a positive finite sum")
@icontract.require(lambda n_samples: _positive_int(n_samples), "n_samples must be positive")
@icontract.ensure(
    lambda result, n_samples: isinstance(result, np.ndarray)
    and np.isclose(float(np.sum(result)), float(n_samples)),
    "rescaled sample_weight must sum to n_samples",
)
def cd_estimator_sample_weight_rescaled(
    sample_weight: NDArray[np.floating], n_samples: int
) -> NDArray[np.floating]:
    """Return sample_weight rescaled by n_samples / np.sum(sample_weight)."""
    array = np.asarray(sample_weight)
    return array * (int(n_samples) / np.sum(array))
