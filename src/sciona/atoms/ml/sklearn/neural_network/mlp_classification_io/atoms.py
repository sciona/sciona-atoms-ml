"""Classifier-side MLP helper atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray
from sklearn.preprocessing._label import _inverse_binarize_multiclass, _inverse_binarize_thresholding
from sklearn.utils.multiclass import type_of_target, unique_labels

from sciona.ghost.registry import register_atom

from sciona.atoms.ml.sklearn.preprocessing.atoms import label_binarizer_fit, label_binarizer_transform
from sciona.atoms.ml.sklearn.preprocessing.state_models import LabelBinarizerState

from .witnesses import (
    witness_mlp_classifier_encode_targets,
    witness_mlp_classifier_fit_target_state,
    witness_mlp_classifier_labels_from_outputs,
    witness_mlp_classifier_partial_fit_target_state,
    witness_mlp_classifier_probabilities_from_outputs,
)

LabelArray = NDArray[np.object_] | NDArray[np.int64] | NDArray[np.float64] | NDArray[np.bool_]
DecodedLabels = NDArray[np.object_] | sp.csr_matrix

_LABEL_TYPES = {"binary", "multiclass", "multilabel-indicator"}


def _label_input_valid(y: LabelArray) -> bool:
    try:
        array = np.asarray(y)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim in {1, 2} and array.shape[0] >= 1 and (array.ndim == 1 or array.shape[1] >= 1))


def _output_valid(outputs: NDArray[np.float64]) -> bool:
    try:
        values = np.asarray(outputs, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(values.ndim in {1, 2} and values.shape[0] >= 1 and np.all(np.isfinite(values)))


def _state_valid(state: LabelBinarizerState | None) -> bool:
    if state is None:
        return True
    classes = np.asarray(state.classes)
    return bool(
        classes.ndim == 1
        and classes.shape[0] >= 1
        and isinstance(state.y_type, str)
        and state.y_type in _LABEL_TYPES
        and isinstance(state.sparse_input, bool)
        and isinstance(state.neg_label, int)
        and isinstance(state.pos_label, int)
        and state.neg_label < state.pos_label
        and isinstance(state.sparse_output, bool)
    )


def _fitted_state_valid(state: LabelBinarizerState) -> bool:
    return bool(_state_valid(state) and state is not None)


def _positive_int(value: int) -> bool:
    return bool(isinstance(value, int) and not isinstance(value, bool) and value >= 1)


def _fit_combo_valid(existing_state: LabelBinarizerState | None, incremental: bool) -> bool:
    return bool(not incremental or existing_state is not None)


def _encoded_targets_valid(result: NDArray[np.bool_], y: LabelArray, state: LabelBinarizerState) -> bool:
    values = np.asarray(result)
    rows = np.asarray(y).shape[0]
    width = np.asarray(state.classes).shape[0] if np.asarray(state.classes).shape[0] > 2 else 1
    return bool(values.shape == (rows, width) and values.dtype == np.bool_)


def _decoded_labels_valid(result: DecodedLabels, outputs: NDArray[np.float64]) -> bool:
    rows = np.asarray(outputs).shape[0]
    if sp.issparse(result):
        values = result.tocsr()
        return bool(values.shape[0] == rows)
    return bool(np.asarray(result).shape[0] == rows)


def _probabilities_valid(result: NDArray[np.float64], outputs: NDArray[np.float64], n_outputs: int) -> bool:
    values = np.asarray(result, dtype=np.float64)
    source = np.asarray(outputs, dtype=np.float64)
    if not (values.ndim == 2 and values.shape[0] == source.shape[0] and np.all(np.isfinite(values))):
        return False
    if not (np.all(values >= 0.0) and np.all(values <= 1.0)):
        return False
    if n_outputs == 1:
        return bool(values.shape[1] == 2 and np.allclose(values.sum(axis=1), 1.0))
    return bool(source.ndim == 2 and values.shape == source.shape)


def _inverse_from_state(outputs: NDArray[np.float64], state: LabelBinarizerState) -> DecodedLabels:
    threshold = (state.pos_label + state.neg_label) / 2.0
    if state.y_type == "multiclass":
        decoded = _inverse_binarize_multiclass(outputs, state.classes)
    else:
        decoded = _inverse_binarize_thresholding(outputs, state.y_type, state.classes, threshold)
    if state.sparse_input:
        return sp.csr_matrix(decoded)
    if sp.issparse(decoded):
        return decoded.toarray()
    return np.asarray(decoded, dtype=object)


@register_atom(witness_mlp_classifier_fit_target_state)
@icontract.require(lambda y: _label_input_valid(y), "y must be a nonempty 1D label vector or 2D multilabel indicator")
@icontract.require(lambda existing_state: _state_valid(existing_state), "existing_state must be a valid fitted label-binarizer state when provided")
@icontract.require(lambda existing_state, incremental: _fit_combo_valid(existing_state, incremental), "incremental fit requires an existing fitted label state")
@icontract.ensure(lambda result: _fitted_state_valid(result), "fit target state must be a valid label-binarizer state")
def mlp_classifier_fit_target_state(
    y: LabelArray,
    *,
    existing_state: LabelBinarizerState | None = None,
    warm_start: bool = False,
    incremental: bool = False,
) -> LabelBinarizerState:
    """Resolve the fitted label-binarizer state sklearn's MLPClassifier uses during fit."""
    if existing_state is None or (not warm_start and not incremental):
        return label_binarizer_fit(y)

    classes = unique_labels(y)
    if warm_start:
        if set(classes) != set(existing_state.classes):
            raise ValueError(
                "warm_start can only be used where `y` has the same classes as in the previous call to fit. "
                f"Previously got {existing_state.classes}, `y` has {classes}"
            )
    elif len(np.setdiff1d(classes, existing_state.classes, assume_unique=True)):
        raise ValueError(
            "`y` has classes not in `self.classes_`. "
            f"`self.classes_` has {existing_state.classes}. 'y' has {classes}."
        )
    return existing_state


@register_atom(witness_mlp_classifier_partial_fit_target_state)
@icontract.require(lambda y: _label_input_valid(y), "y must be a nonempty 1D label vector or 2D multilabel indicator")
@icontract.require(lambda classes: _label_input_valid(classes) and np.asarray(classes).ndim == 1, "classes must be a nonempty 1D class vector")
@icontract.ensure(lambda result: _fitted_state_valid(result), "partial-fit target state must be a valid label-binarizer state")
def mlp_classifier_partial_fit_target_state(
    y: LabelArray,
    classes: LabelArray,
) -> LabelBinarizerState:
    """Build the first-call label-binarizer state sklearn's MLPClassifier.partial_fit uses."""
    if type_of_target(y).startswith("multilabel"):
        return label_binarizer_fit(y)
    return label_binarizer_fit(classes)


@register_atom(witness_mlp_classifier_encode_targets)
@icontract.require(lambda y: _label_input_valid(y), "y must be a nonempty 1D label vector or 2D multilabel indicator")
@icontract.require(lambda state: _fitted_state_valid(state), "state must be a valid label-binarizer state")
@icontract.ensure(lambda result, y, state: _encoded_targets_valid(result, y, state), "encoded targets must be a boolean matrix with sklearn-compatible width")
def mlp_classifier_encode_targets(
    y: LabelArray,
    state: LabelBinarizerState,
) -> NDArray[np.bool_]:
    """Transform classifier targets to sklearn's boolean MLP training matrix."""
    encoded = label_binarizer_transform(y, state)
    if sp.issparse(encoded):
        encoded = encoded.toarray()
    return np.asarray(encoded, dtype=np.bool_)


@register_atom(witness_mlp_classifier_labels_from_outputs)
@icontract.require(lambda outputs: _output_valid(outputs), "outputs must be finite 1D or 2D classifier output scores")
@icontract.require(lambda state: _fitted_state_valid(state), "state must be a valid label-binarizer state")
@icontract.require(lambda n_outputs: _positive_int(n_outputs), "n_outputs must be a positive integer")
@icontract.ensure(lambda result, outputs: _decoded_labels_valid(result, outputs), "decoded labels must preserve the sample count")
def mlp_classifier_labels_from_outputs(
    outputs: NDArray[np.float64],
    state: LabelBinarizerState,
    *,
    n_outputs: int,
) -> DecodedLabels:
    """Decode already-computed MLP classifier outputs back to predicted labels."""
    values = np.asarray(outputs, dtype=np.float64)
    if n_outputs == 1:
        return _inverse_from_state(values.ravel(), state)
    return _inverse_from_state(values, state)


@register_atom(witness_mlp_classifier_probabilities_from_outputs)
@icontract.require(lambda outputs: _output_valid(outputs), "outputs must be finite 1D or 2D classifier output probabilities")
@icontract.require(lambda n_outputs: _positive_int(n_outputs), "n_outputs must be a positive integer")
@icontract.ensure(lambda result, outputs, n_outputs: _probabilities_valid(result, outputs, n_outputs), "probability output must match sklearn's binary stacking or multiclass passthrough")
def mlp_classifier_probabilities_from_outputs(
    outputs: NDArray[np.float64],
    *,
    n_outputs: int,
) -> NDArray[np.float64]:
    """Format already-computed MLP classifier outputs as predict_proba returns them."""
    values = np.asarray(outputs, dtype=np.float64)
    if n_outputs == 1:
        flattened = values.ravel()
        return np.vstack([1.0 - flattened, flattened]).T
    return values.copy()
