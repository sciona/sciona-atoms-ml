"""Ghost witnesses for sklearn MLP classifier I/O helper atoms."""

from __future__ import annotations

from scipy.sparse import csr_matrix

from sciona.ghost.abstract import AbstractArray

from sciona.atoms.ml.sklearn.preprocessing.state_models import LabelBinarizerState


def _check_label_input(values: AbstractArray, name: str) -> tuple[int, int]:
    if len(values.shape) not in {1, 2}:
        raise ValueError(f"{name} must be 1D or 2D")
    rows = int(values.shape[0])
    if rows < 1:
        raise ValueError(f"{name} must be nonempty")
    cols = 1 if len(values.shape) == 1 else int(values.shape[1])
    if cols < 1:
        raise ValueError(f"{name} must have at least one column")
    return rows, cols


def _check_output_matrix(values: AbstractArray, name: str) -> tuple[int, int]:
    if len(values.shape) not in {1, 2}:
        raise ValueError(f"{name} must be 1D or 2D")
    rows = int(values.shape[0])
    if rows < 1:
        raise ValueError(f"{name} must be nonempty")
    cols = 1 if len(values.shape) == 1 else int(values.shape[1])
    if cols < 1:
        raise ValueError(f"{name} must have at least one output column")
    return rows, cols


def witness_mlp_classifier_fit_target_state(
    y: AbstractArray,
    *,
    existing_state: LabelBinarizerState | None = None,
    warm_start: bool = False,
    incremental: bool = False,
) -> LabelBinarizerState:
    """Describe the fitted label-binarizer state used by MLPClassifier fit."""
    del warm_start, incremental
    _check_label_input(y, "y")
    if existing_state is not None and existing_state.classes.shape[0] < 1:
        raise ValueError("existing_state.classes must be nonempty")
    return LabelBinarizerState(
        classes=AbstractArray(shape=(1,), dtype="object"),
        y_type="binary",
        sparse_input=False,
        neg_label=0,
        pos_label=1,
        sparse_output=False,
    )


def witness_mlp_classifier_partial_fit_target_state(
    y: AbstractArray,
    classes: AbstractArray,
) -> LabelBinarizerState:
    """Describe the initial label-binarizer state used by MLPClassifier.partial_fit."""
    _check_label_input(y, "y")
    if len(classes.shape) != 1 or int(classes.shape[0]) < 1:
        raise ValueError("classes must be a nonempty vector")
    return LabelBinarizerState(
        classes=AbstractArray(shape=(1,), dtype="object"),
        y_type="binary",
        sparse_input=False,
        neg_label=0,
        pos_label=1,
        sparse_output=False,
    )


def witness_mlp_classifier_encode_targets(
    y: AbstractArray,
    state: LabelBinarizerState,
) -> AbstractArray:
    """Describe boolean MLP classifier targets encoded from fitted label state."""
    rows, _ = _check_label_input(y, "y")
    width = int(state.classes.shape[0]) if int(state.classes.shape[0]) > 2 else 1
    return AbstractArray(shape=(rows, width), dtype="bool")


def witness_mlp_classifier_labels_from_outputs(
    outputs: AbstractArray,
    state: LabelBinarizerState,
    *,
    n_outputs: int,
) -> AbstractArray | csr_matrix:
    """Turn output scores into labels."""
    rows, _ = _check_output_matrix(outputs, "outputs")
    del n_outputs
    if state.sparse_input:
        return csr_matrix((rows, int(state.classes.shape[0]) if state.y_type.startswith("multilabel") else 1), dtype=int)
    return AbstractArray(shape=(rows,), dtype="object")


def witness_mlp_classifier_probabilities_from_outputs(
    outputs: AbstractArray,
    *,
    n_outputs: int,
) -> AbstractArray:
    """Describe probability output formatting for MLPClassifier predict_proba."""
    rows, cols = _check_output_matrix(outputs, "outputs")
    if n_outputs < 1:
        raise ValueError("n_outputs must be positive")
    if n_outputs == 1:
        return AbstractArray(shape=(rows, 2), dtype="float64")
    return AbstractArray(shape=(rows, cols), dtype="float64")
