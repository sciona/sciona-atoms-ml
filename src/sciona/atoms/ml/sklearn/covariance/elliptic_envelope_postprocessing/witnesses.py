"""Witnesses for sklearn EllipticEnvelope postprocessing helpers."""

from __future__ import annotations

import numpy as np

from sciona.ghost.abstract import AbstractArray


def _check_vector(values: AbstractArray, name: str) -> int:
    if len(values.shape) != 1:
        raise ValueError(f"{name} must be 1D")
    n_samples = int(values.shape[0])
    if n_samples < 1:
        raise ValueError(f"{name} must be nonempty")
    return n_samples


def witness_elliptic_envelope_offset(
    negative_training_distances: AbstractArray,
    contamination: float,
) -> float:
    """Describe the scalar fitted offset computed from training score samples."""
    _check_vector(negative_training_distances, "negative_training_distances")
    return float(contamination)


def witness_elliptic_envelope_score_samples(
    mahalanobis_distances: AbstractArray,
) -> AbstractArray:
    """Describe the score_samples sign flip from Mahalanobis distances."""
    _check_vector(mahalanobis_distances, "mahalanobis_distances")
    return AbstractArray(shape=mahalanobis_distances.shape, dtype=np.float64)


def witness_elliptic_envelope_decision_function(
    negative_mahalanobis_distances: AbstractArray,
    offset: float,
) -> AbstractArray:
    """Describe the decision-function shift from score samples and offset."""
    del offset
    _check_vector(negative_mahalanobis_distances, "negative_mahalanobis_distances")
    return AbstractArray(shape=negative_mahalanobis_distances.shape, dtype=np.float64)


def witness_elliptic_envelope_labels(
    decision_values: AbstractArray,
) -> AbstractArray:
    """Describe the {-1, 1} label vector produced from decision values."""
    _check_vector(decision_values, "decision_values")
    return AbstractArray(shape=decision_values.shape, dtype=np.int64)
