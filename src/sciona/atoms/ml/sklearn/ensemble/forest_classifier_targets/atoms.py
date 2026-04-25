"""Functions for forest classifier targets and class weights."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import icontract
import numpy as np
from numpy.typing import NDArray
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.utils.multiclass import check_classification_targets

from sciona.ghost.registry import register_atom

from sciona.atoms.ml.sklearn.ensemble.state_models import ForestClassifierTargetState

from .witnesses import (
    witness_forest_classifier_class_weight_warning_required,
    witness_forest_classifier_expanded_class_weight,
    witness_forest_classifier_fit_targets,
    witness_forest_classifier_validate_class_weight_preset,
)

FitTargetsResult = tuple[ForestClassifierTargetState, NDArray[np.int64]]


def _target_matrix_valid(y: object) -> bool:
    try:
        values = np.asarray(y)
    except (TypeError, ValueError):
        return False
    return bool(values.ndim == 2 and values.shape[0] >= 1 and values.shape[1] >= 1)


def _state_valid(state: ForestClassifierTargetState, n_outputs: int) -> bool:
    return bool(
        isinstance(state.classes, tuple)
        and isinstance(state.n_classes, tuple)
        and len(state.classes) == n_outputs
        and len(state.n_classes) == n_outputs
        and all(
            np.asarray(classes, dtype=object).ndim == 1
            and np.asarray(classes, dtype=object).shape[0] >= 1
            and np.unique(np.asarray(classes, dtype=object)).shape[0]
            == np.asarray(classes, dtype=object).shape[0]
            for classes in state.classes
        )
        and all(
            isinstance(count, int)
            and count == int(np.asarray(classes, dtype=object).shape[0])
            for classes, count in zip(state.classes, state.n_classes)
        )
    )


def _fit_targets_result_valid(result: FitTargetsResult, y: object) -> bool:
    state, encoded = result
    matrix = np.asarray(y)
    values = np.asarray(encoded)
    if not (
        _state_valid(state, int(matrix.shape[1]))
        and values.shape == matrix.shape
        and np.issubdtype(values.dtype, np.integer)
        and np.all(values >= 0)
    ):
        return False
    return all(np.all(values[:, column] < state.n_classes[column]) for column in range(values.shape[1]))


def _preset_input_valid(class_weight: object) -> bool:
    return class_weight is None or isinstance(class_weight, str)


def _preset_output_valid(result: object) -> bool:
    return result is None or result in {"balanced", "balanced_subsample"}


def _class_weight_object_valid(class_weight: object, n_outputs: int) -> bool:
    if class_weight is None:
        return True
    if isinstance(class_weight, str):
        return class_weight in {"balanced", "balanced_subsample"}
    if isinstance(class_weight, Mapping):
        return True
    if isinstance(class_weight, Sequence) and not isinstance(class_weight, (str, bytes)):
        return len(class_weight) == n_outputs and all(isinstance(item, Mapping) for item in class_weight)
    return False


def _target_output_count(y: object) -> int:
    try:
        values = np.asarray(y)
    except (TypeError, ValueError):
        return 0
    if values.ndim != 2:
        return 0
    return int(values.shape[1])


def _expanded_class_weight_valid(result: object, y_original: object) -> bool:
    if result is None:
        return True
    try:
        values = np.asarray(result, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    y_matrix = np.asarray(y_original)
    return bool(
        values.shape == (y_matrix.shape[0],)
        and np.all(np.isfinite(values))
        and np.all(values > 0.0)
    )


@register_atom(witness_forest_classifier_fit_targets)
@icontract.require(
    lambda y: _target_matrix_valid(y),
    "y must be a nonempty 2D target matrix",
)
@icontract.ensure(
    lambda result, y: _fit_targets_result_valid(result, y),
    "fit targets result must contain per-output class state and encoded targets",
)
def forest_classifier_fit_targets(
    y: NDArray[np.object_] | NDArray[np.int64] | NDArray[np.float64] | NDArray[np.bool_],
) -> FitTargetsResult:
    """Resolve forest classifier classes, class counts, and encoded targets."""
    matrix = np.asarray(y).copy()
    check_classification_targets(matrix)

    classes: list[NDArray[np.object_]] = []
    n_classes: list[int] = []
    encoded = np.zeros(matrix.shape, dtype=np.int64)
    for column in range(matrix.shape[1]):
        classes_column, encoded[:, column] = np.unique(matrix[:, column], return_inverse=True)
        classes.append(np.asarray(classes_column, dtype=object))
        n_classes.append(int(classes_column.shape[0]))

    state = ForestClassifierTargetState(
        classes=tuple(classes),
        n_classes=tuple(n_classes),
    )
    return state, np.asarray(encoded, dtype=np.int64)


@register_atom(witness_forest_classifier_validate_class_weight_preset)
@icontract.require(
    lambda class_weight: _preset_input_valid(class_weight),
    "class_weight preset input must be None or a string",
)
@icontract.ensure(
    lambda result: _preset_output_valid(result),
    "validated class_weight preset must be None, balanced, or balanced_subsample",
)
def forest_classifier_validate_class_weight_preset(
    class_weight: str | None,
) -> str | None:
    """Validate sklearn's forest classifier string presets for class_weight."""
    if class_weight is None:
        return None
    if class_weight not in {"balanced", "balanced_subsample"}:
        raise ValueError(
            "Valid presets for class_weight include "
            '"balanced" and "balanced_subsample".'
            f'Given "{class_weight}".'
        )
    return class_weight


@register_atom(witness_forest_classifier_class_weight_warning_required)
@icontract.require(
    lambda class_weight: _preset_input_valid(class_weight),
    "class_weight must be None or a string for warning gating",
)
@icontract.ensure(lambda result: isinstance(result, bool), "warning-required result must be boolean")
def forest_classifier_class_weight_warning_required(
    class_weight: str | None,
    warm_start: bool,
) -> bool:
    """Return whether sklearn would issue the warm-start class-weight warning."""
    preset = forest_classifier_validate_class_weight_preset(class_weight)
    return bool(preset is not None and warm_start)


@register_atom(witness_forest_classifier_expanded_class_weight)
@icontract.require(
    lambda y_original: _target_matrix_valid(y_original),
    "y_original must be a nonempty 2D target matrix",
)
@icontract.require(
    lambda y_original, class_weight: _class_weight_object_valid(class_weight, _target_output_count(y_original)),
    "class_weight must be None, a valid preset string, a mapping, or one mapping per output",
)
@icontract.ensure(
    lambda result, y_original: _expanded_class_weight_valid(result, y_original),
    "expanded class weight must be None or a positive finite sample-weight vector",
)
def forest_classifier_expanded_class_weight(
    y_original: NDArray[np.object_] | NDArray[np.int64] | NDArray[np.float64] | NDArray[np.bool_],
    class_weight: str | Mapping[object, float] | Sequence[Mapping[object, float]] | None,
    *,
    bootstrap: bool,
) -> NDArray[np.float64] | None:
    """Resolve forest classifier expanded sample weights from original targets."""
    matrix = np.asarray(y_original).copy()
    preset = class_weight if isinstance(class_weight, str) else None
    if preset is not None:
        forest_classifier_validate_class_weight_preset(preset)
    if class_weight is None:
        return None
    if class_weight == "balanced_subsample" and bootstrap:
        return None
    effective_class_weight: object
    if class_weight == "balanced_subsample":
        effective_class_weight = "balanced"
    else:
        effective_class_weight = class_weight
    return np.asarray(compute_sample_weight(effective_class_weight, matrix), dtype=np.float64)
