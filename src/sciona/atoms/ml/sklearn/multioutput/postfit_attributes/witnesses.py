"""Witnesses for sklearn multioutput post-fit attribute helpers."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from sciona.ghost.abstract import AbstractArray

ClassVectorTuple = tuple[NDArray[np.float64], ...]


def witness_multioutput_partial_fit_n_features_in_update_required(
    *,
    first_time: bool,
    estimator_has_n_features_in: bool,
) -> bool:
    """Describe the boolean partial-fit n_features_in_ update condition."""
    return bool(first_time and estimator_has_n_features_in)


def witness_multioutput_partial_fit_feature_names_in_update_required(
    *,
    first_time: bool,
    estimator_has_feature_names_in: bool,
) -> bool:
    """Describe the boolean partial-fit feature_names_in_ update condition."""
    return bool(first_time and estimator_has_feature_names_in)


def witness_multioutput_fit_n_features_in(n_features_in: int) -> int:
    """Describe the fitted feature-count attribute pass-through."""
    return int(n_features_in)


def witness_multioutput_fit_feature_names_in(feature_names_in: tuple[str, ...]) -> tuple[str, ...]:
    """Describe the fitted feature-name tuple pass-through."""
    return tuple(feature_names_in)


def witness_multioutput_classifier_classes(
    class_vectors: tuple[AbstractArray, ...],
) -> tuple[AbstractArray, ...]:
    """Describe the per-estimator class-vector tuple pass-through."""
    return tuple(class_vectors)
