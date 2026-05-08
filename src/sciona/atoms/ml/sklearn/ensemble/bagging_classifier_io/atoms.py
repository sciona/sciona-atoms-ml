"""Bagging classifier I/O helpers adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from sciona.atoms.ml.sklearn.ensemble.state_models import BaggingClassifierTargetState

from .witnesses import (
    witness_bagging_classifier_fit_targets,
    witness_bagging_classifier_labels_from_probabilities,
)

FitTargetsResult = tuple[BaggingClassifierTargetState, NDArray[np.int64]]

def _label_input_valid(y: object) -> bool:
    try:
        values = np.asarray(y)
    except (TypeError, ValueError):
        return False
    return bool(
        values.ndim in {1, 2}
        and values.shape[0] >= 1
        and (values.ndim == 1 or values.shape[1] == 1)
    )

def _state_valid(state: BaggingClassifierTargetState) -> bool:
    classes = np.asarray(state.classes, dtype=object)
    return bool(
        classes.ndim == 1
        and classes.shape[0] >= 1
        and np.unique(classes).shape[0] == classes.shape[0]
        and isinstance(state.n_classes, int)
        and state.n_classes == int(classes.shape[0])
    )

def _fit_targets_result_valid(result: FitTargetsResult, y: object) -> bool:
    state, encoded = result
    values = np.asarray(encoded)
    rows = int(np.asarray(y).shape[0])
    return bool(
        _state_valid(state)
        and values.shape == (rows,)
        and np.issubdtype(values.dtype, np.integer)
        and np.all(values >= 0)
        and np.all(values < state.n_classes)
    )

def _classes_valid(classes: object) -> bool:
    values = np.asarray(classes, dtype=object)
    return bool(values.ndim == 1 and values.shape[0] >= 1 and np.unique(values).shape[0] == values.shape[0])

def _aggregated_probabilities_valid(probabilities: object, classes: object) -> bool:
    try:
        values = np.asarray(probabilities, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    class_values = np.asarray(classes, dtype=object)
    return bool(
        values.ndim == 2
        and values.shape[0] >= 1
        and values.shape[1] >= 1
        and np.all(np.isfinite(values))
        and np.all(values >= 0.0)
        and np.allclose(np.sum(values, axis=1), 1.0)
        and _classes_valid(classes)
        and values.shape[1] == class_values.shape[0]
    )

def _labels_valid(result: object, probabilities: object, classes: object) -> bool:
    values = np.asarray(result, dtype=object)
    probability_values = np.asarray(probabilities, dtype=np.float64)
    class_values = np.asarray(classes, dtype=object)
    return bool(values.shape == (probability_values.shape[0],) and np.isin(values, class_values).all())

@register_atom(witness_bagging_classifier_fit_targets)
@icontract.require(lambda y: _label_input_valid(y), "y must be a nonempty 1D label vector or a single-column 2D label array")
@icontract.ensure(lambda result, y: _fit_targets_result_valid(result, y), "fit targets result must contain unique classes and encoded labels for each sample")
def bagging_classifier_fit_targets(
    y: NDArray[np.object_] | NDArray[np.int64] | NDArray[np.float64] | NDArray[np.bool_],
) -> FitTargetsResult:
    from sklearn.utils import column_or_1d
    from sklearn.utils.multiclass import check_classification_targets
    """Resolve sklearn bagging classifier classes and encoded targets during fit."""
    checked = column_or_1d(y, warn=True)
    check_classification_targets(checked)
    classes, encoded = np.unique(checked, return_inverse=True)
    state = BaggingClassifierTargetState(
        classes=np.asarray(classes, dtype=object),
        n_classes=int(classes.shape[0]),
    )
    return state, np.asarray(encoded, dtype=np.int64)

@register_atom(witness_bagging_classifier_labels_from_probabilities)
@icontract.require(lambda probabilities, classes: _aggregated_probabilities_valid(probabilities, classes), "probabilities must be a normalized sample-by-class matrix matching the classes vector")
@icontract.ensure(lambda result, probabilities, classes: _labels_valid(result, probabilities, classes), "predicted labels must come from the classes vector")
def bagging_classifier_labels_from_probabilities(
    probabilities: NDArray[np.float64],
    classes: NDArray[np.object_] | NDArray[np.int64] | NDArray[np.float64] | NDArray[np.bool_],
) -> NDArray[np.object_]:
    """Decode bagging classifier labels from already-aggregated class probabilities."""
    probability_values = np.asarray(probabilities, dtype=np.float64)
    class_values = np.asarray(classes, dtype=object)
    return np.asarray(class_values.take(np.argmax(probability_values, axis=1), axis=0), dtype=object)
