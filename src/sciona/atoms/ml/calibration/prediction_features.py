"""Explicit numerical boundaries for residual-classifier compositions."""
import numpy as np
from numpy.typing import NDArray
from sciona.ghost.abstract import AbstractArray
from sciona.ghost.registry import register_atom


def _vector(value, dtypes):
    result = np.asarray(value)
    if result.ndim != 1 or result.dtype not in dtypes or not np.isfinite(result).all():
        raise ValueError('Finite vector with a supported prediction dtype required')
    return result


def witness_round_to_int32(values: AbstractArray) -> AbstractArray:
    if len(values.shape) != 1 or values.dtype != 'float32':
        raise ValueError('Float32 prediction vector metadata required')
    return AbstractArray(shape=values.shape, dtype='int32', dim=values.dim)


@register_atom(witness_round_to_int32)
def round_to_int32(values: NDArray[np.float32]) -> NDArray[np.int32]:
    """Round float32 predictions to nearest integer, ties to even, then cast.

    Quantization step is one in caller-declared target units. Rounded values
    outside int32 range fail explicitly instead of relying on invalid casts.
    No clipping or conversion of physical units. Empty vectors remain empty.
    """
    array = _vector(values, (np.dtype('float32'),))
    rounded = np.around(array).astype(np.float64)
    bounds = np.iinfo(np.int32)
    if ((rounded < bounds.min) | (rounded > bounds.max)).any():
        raise ValueError('Rounded predictions outside int32 range')
    return rounded.astype(np.int32)


def witness_widen_prediction_values(values: AbstractArray) -> AbstractArray:
    if len(values.shape) != 1 or values.dtype not in ('int32', 'float32'):
        raise ValueError('Int32 or float32 vector metadata required')
    return AbstractArray(shape=values.shape, dtype='float64', dim=values.dim)


@register_atom(witness_widen_prediction_values)
def widen_prediction_values(values: NDArray) -> NDArray[np.float64]:
    """Exactly widen int32 values or finite float32 values to independent float64."""
    return _vector(values, (np.dtype('int32'), np.dtype('float32'))).astype(np.float64)


def witness_underestimation_labels(observed: AbstractArray, predictions: AbstractArray) -> AbstractArray:
    if len(observed.shape) != 1 or observed.shape != predictions.shape or observed.dtype != 'float64' or predictions.dtype != 'int32':
        raise ValueError('Aligned float64 observations and int32 predictions required')
    if observed.dim is not None and predictions.dim is not None and not observed.dim.is_compatible(predictions.dim):
        raise ValueError('Observations and predictions must share target units')
    return AbstractArray(shape=observed.shape, dtype='int64')


@register_atom(witness_underestimation_labels)
def underestimation_labels(observed: NDArray[np.float64], predictions: NDArray[np.int32]) -> NDArray[np.int64]:
    """Label strict underprediction as 1, and overprediction or equality as 0.

    Observations and already-quantized predictions share target units. This
    computes target labels; it does not fit or assess a classifier.
    """
    target = _vector(observed, (np.dtype('float64'),))
    prediction = _vector(predictions, (np.dtype('int32'),))
    if target.shape != prediction.shape:
        raise ValueError('Aligned observations and predictions required')
    return (prediction < target).astype(np.int64)


def witness_append_prediction_feature(features: AbstractArray, predictions: AbstractArray) -> AbstractArray:
    if len(features.shape) != 2 or features.dtype != 'float64' or predictions.shape != (features.shape[0],) or predictions.dtype not in ('int32', 'float32'):
        raise ValueError('Float64 matrix and aligned prediction vector metadata required')
    # Heterogeneous column units must be retained in the caller's ordered schema.
    return AbstractArray(shape=(features.shape[0], features.shape[1]+1), dtype='float64')


@register_atom(witness_append_prediction_feature)
def append_prediction_feature(features: NDArray[np.float64], predictions: NDArray) -> NDArray[np.float64]:
    """Append one explicitly ordered prediction column to a float64 matrix.

    Int32 training predictions and float32 inference predictions are widened
    exactly. No inference rounding is introduced. Existing feature NaNs retain
    backend missing-value semantics; infinities are rejected. The caller appends
    the prediction column's unique name and target units to its feature schema.
    """
    matrix = np.asarray(features)
    prediction = _vector(predictions, (np.dtype('int32'), np.dtype('float32')))
    if matrix.dtype != np.float64 or matrix.ndim != 2 or len(matrix) != len(prediction) or np.isinf(matrix).any():
        raise ValueError('Aligned float64 feature matrix without infinities required')
    return np.column_stack((matrix, prediction.astype(np.float64)))
