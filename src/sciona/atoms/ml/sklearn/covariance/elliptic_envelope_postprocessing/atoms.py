"""EllipticEnvelope postprocessing helpers adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_elliptic_envelope_decision_function,
    witness_elliptic_envelope_labels,
    witness_elliptic_envelope_offset,
    witness_elliptic_envelope_score_samples,
)


def _distance_vector_valid(values: object) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 1 and array.shape[0] >= 1 and np.all(np.isfinite(array)) and np.all(array >= 0.0))


def _score_vector_valid(values: object) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 1 and array.shape[0] >= 1 and np.all(np.isfinite(array)))


def _contamination_valid(value: object) -> bool:
    return bool(isinstance(value, float) and np.isfinite(value) and 0.0 < value <= 0.5)


def _finite_scalar(value: object) -> bool:
    return bool(isinstance(value, float) and np.isfinite(value))


def _same_shape_vector(result: object, source: object) -> bool:
    try:
        observed = np.asarray(result)
        expected = np.asarray(source)
    except (TypeError, ValueError):
        return False
    return bool(observed.ndim == 1 and expected.ndim == 1 and observed.shape == expected.shape)


def _offset_valid(result: object) -> bool:
    return _finite_scalar(result)


def _labels_valid(result: object, decision_values: object) -> bool:
    if not _same_shape_vector(result, decision_values):
        return False
    values = np.asarray(result)
    return bool(np.issubdtype(values.dtype, np.integer) and set(np.unique(values)).issubset({-1, 1}))


@register_atom(witness_elliptic_envelope_offset)
@icontract.require(lambda negative_training_distances: _score_vector_valid(negative_training_distances), "negative_training_distances must be a nonempty finite 1D vector")
@icontract.require(lambda contamination: _contamination_valid(contamination), "contamination must lie in (0, 0.5]")
@icontract.ensure(lambda result: _offset_valid(result), "offset must be finite")
def elliptic_envelope_offset(
    negative_training_distances: NDArray[np.float64],
    contamination: float,
) -> float:
    """Compute sklearn's fitted EllipticEnvelope offset from negative training distances."""
    return float(np.percentile(np.asarray(negative_training_distances, dtype=np.float64), 100.0 * contamination))


@register_atom(witness_elliptic_envelope_score_samples)
@icontract.require(lambda mahalanobis_distances: _distance_vector_valid(mahalanobis_distances), "mahalanobis_distances must be a nonempty finite nonnegative 1D vector")
@icontract.ensure(lambda result, mahalanobis_distances: _same_shape_vector(result, mahalanobis_distances), "score_samples must preserve the input shape")
def elliptic_envelope_score_samples(
    mahalanobis_distances: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Convert supplied Mahalanobis distances to sklearn's negative score_samples output."""
    return -np.asarray(mahalanobis_distances, dtype=np.float64)


@register_atom(witness_elliptic_envelope_decision_function)
@icontract.require(lambda negative_mahalanobis_distances: _score_vector_valid(negative_mahalanobis_distances), "negative_mahalanobis_distances must be a nonempty finite 1D vector")
@icontract.require(lambda offset: _finite_scalar(offset), "offset must be finite")
@icontract.ensure(lambda result, negative_mahalanobis_distances: _same_shape_vector(result, negative_mahalanobis_distances), "decision_function must preserve the score vector shape")
def elliptic_envelope_decision_function(
    negative_mahalanobis_distances: NDArray[np.float64],
    offset: float,
) -> NDArray[np.float64]:
    """Shift negative Mahalanobis scores by sklearn's fitted EllipticEnvelope offset."""
    return np.asarray(np.asarray(negative_mahalanobis_distances, dtype=np.float64) - float(offset), dtype=np.float64)


@register_atom(witness_elliptic_envelope_labels)
@icontract.require(lambda decision_values: _score_vector_valid(decision_values), "decision_values must be a nonempty finite 1D vector")
@icontract.ensure(lambda result, decision_values: _labels_valid(result, decision_values), "labels must preserve sample count and contain only -1 or 1")
def elliptic_envelope_labels(
    decision_values: NDArray[np.float64],
) -> NDArray[np.int64]:
    """Threshold EllipticEnvelope decision values into sklearn's {-1, 1} outlier labels."""
    values = np.asarray(decision_values, dtype=np.float64)
    labels = np.full(values.shape[0], -1, dtype=np.int64)
    labels[values >= 0.0] = 1
    return labels
