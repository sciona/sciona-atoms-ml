"""One-vs-one prediction-output helpers adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_one_vs_one_binary_labels,
    witness_one_vs_one_multiclass_labels,
)


def _classes_valid(classes: object, *, min_classes: int = 1) -> bool:
    try:
        values = np.asarray(classes, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(
        values.ndim == 1
        and values.shape[0] >= min_classes
        and np.all(np.isfinite(values))
        and np.unique(values).shape[0] == values.shape[0]
    )


def _binary_scores_valid(scores: object) -> bool:
    try:
        values = np.asarray(scores, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(values.ndim == 1 and values.shape[0] >= 1 and np.all(np.isfinite(values)))


def _multiclass_scores_valid(scores: object, classes: object) -> bool:
    try:
        values = np.asarray(scores, dtype=np.float64)
        class_values = np.asarray(classes, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(
        values.ndim == 2
        and values.shape[0] >= 1
        and values.shape[1] >= 2
        and np.all(np.isfinite(values))
        and _classes_valid(classes, min_classes=2)
        and values.shape[1] == class_values.shape[0]
    )


def _finite_threshold(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(float(value))


def _labels_valid(result: object, n_samples: int, classes: object) -> bool:
    try:
        values = np.asarray(result, dtype=np.float64)
        class_values = np.asarray(classes, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(values.shape == (n_samples,) and np.all(np.isin(values, class_values)))


@register_atom(witness_one_vs_one_binary_labels)
@icontract.require(lambda scores: _binary_scores_valid(scores), "scores must be a finite 1D decision vector")
@icontract.require(lambda classes: _classes_valid(classes, min_classes=2), "classes must be a finite unique class vector with two entries")
@icontract.require(lambda threshold: _finite_threshold(threshold), "threshold must be finite")
@icontract.ensure(lambda result, scores, classes: _labels_valid(result, np.asarray(scores).shape[0], classes), "labels must come from the two-class vector")
def one_vs_one_binary_labels(
    scores: NDArray[np.float64],
    classes: NDArray[np.float64],
    *,
    threshold: float = 0.0,
) -> NDArray[np.float64]:
    """Choose sklearn's binary one-vs-one output labels from a decision vector and threshold."""
    score_values = np.asarray(scores, dtype=np.float64)
    class_values = np.asarray(classes, dtype=np.float64)
    return np.asarray(class_values[(score_values > float(threshold)).astype(int)], dtype=np.float64)


@register_atom(witness_one_vs_one_multiclass_labels)
@icontract.require(lambda scores, classes: _multiclass_scores_valid(scores, classes), "scores must be a finite sample-by-class matrix matching classes")
@icontract.ensure(lambda result, scores, classes: _labels_valid(result, np.asarray(scores).shape[0], classes), "labels must come from the class vector")
def one_vs_one_multiclass_labels(
    scores: NDArray[np.float64],
    classes: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Choose sklearn's multiclass one-vs-one output labels by class-score argmax."""
    score_values = np.asarray(scores, dtype=np.float64)
    class_values = np.asarray(classes, dtype=np.float64)
    return np.asarray(class_values[np.argmax(score_values, axis=1)], dtype=np.float64)
