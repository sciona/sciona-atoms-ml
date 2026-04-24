"""Ghost witnesses for sklearn bagging classifier I/O helpers."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray

from sciona.atoms.ml.sklearn.ensemble.state_models import BaggingClassifierTargetState


def witness_bagging_classifier_fit_targets(
    y: AbstractArray,
) -> tuple[BaggingClassifierTargetState, AbstractArray]:
    """Describe bagging classifier class-state and encoded targets from fit input."""
    if len(y.shape) not in {1, 2}:
        raise ValueError("y must be 1D or 2D")
    rows = int(y.shape[0])
    if rows < 1:
        raise ValueError("y must be nonempty")
    if len(y.shape) == 2 and int(y.shape[1]) != 1:
        raise ValueError("2D y must have exactly one column")
    state = BaggingClassifierTargetState(
        classes=AbstractArray(shape=(1,), dtype="object"),
        n_classes=1,
    )
    return state, AbstractArray(shape=(rows,), dtype="int64", min_val=0.0)


def witness_bagging_classifier_labels_from_probabilities(
    probabilities: AbstractArray,
    classes: AbstractArray,
) -> AbstractArray:
    """Describe label decoding from averaged bagging probabilities."""
    if len(probabilities.shape) != 2:
        raise ValueError("probabilities must be 2D")
    if int(probabilities.shape[0]) < 1 or int(probabilities.shape[1]) < 1:
        raise ValueError("probabilities must be nonempty")
    if len(classes.shape) != 1 or int(classes.shape[0]) < 1:
        raise ValueError("classes must be a nonempty vector")
    return AbstractArray(shape=(int(probabilities.shape[0]),), dtype="object")
