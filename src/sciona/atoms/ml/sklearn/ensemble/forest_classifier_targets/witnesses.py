"""Ghost witnesses for sklearn forest classifier target helpers."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray

from sciona.atoms.ml.sklearn.ensemble.state_models import ForestClassifierTargetState


def witness_forest_classifier_fit_targets(
    y: AbstractArray,
) -> tuple[ForestClassifierTargetState, AbstractArray]:
    """Describe forest classifier target state and encoded targets."""
    if len(y.shape) != 2:
        raise ValueError("y must be a matrix")
    if int(y.shape[0]) < 1 or int(y.shape[1]) < 1:
        raise ValueError("y must be nonempty")
    state = ForestClassifierTargetState(
        classes=(AbstractArray(shape=(1,), dtype="object"),),
        n_classes=(1,),
    )
    return state, AbstractArray(shape=y.shape, dtype="int64", min_val=0.0)


def witness_forest_classifier_validate_class_weight_preset(
    class_weight: object,
) -> object:
    """Describe validation of forest classifier class-weight string presets."""
    return class_weight


def witness_forest_classifier_class_weight_warning_required(
    class_weight: object,
    warm_start: bool,
) -> bool:
    """Describe whether sklearn's warm-start class-weight warning would fire."""
    return bool(warm_start)


def witness_forest_classifier_expanded_class_weight(
    y_original: AbstractArray,
    class_weight: object,
    bootstrap: bool,
) -> AbstractArray | None:
    """Describe expanded per-sample class weights for forest classification."""
    if len(y_original.shape) != 2:
        raise ValueError("y_original must be a matrix")
    if int(y_original.shape[0]) < 1 or int(y_original.shape[1]) < 1:
        raise ValueError("y_original must be nonempty")
    return AbstractArray(shape=(int(y_original.shape[0]),), dtype="float64")
