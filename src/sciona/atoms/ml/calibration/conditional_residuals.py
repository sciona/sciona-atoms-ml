"""Reusable classifier-conditioned median residual correction.

The source example is DrivenData NASA pushback Phase 1, third place. These
atoms expose its numeric correction structure without airport data adapters,
estimator choice, training splits, or submission quantization. Callers must
establish the calibration population's validity and target-unit consistency.
"""
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.abstract import AbstractArray, AbstractScalar
from sciona.ghost.registry import register_atom


def _vector(value):
    array = np.asarray(value)
    if array.dtype != np.float64 or array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ValueError('nonempty finite float64 vector required')
    return array


def _probabilities(value, shape, threshold):
    array = _vector(value)
    if array.shape != shape or np.any((array < 0) | (array > 1)):
        raise ValueError('aligned probabilities in [0,1] required')
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('explicit finite probability threshold required')
    return array


def _metadata(predictions, probabilities, threshold):
    if len(predictions.shape) != 1 or predictions.shape == (0,) or predictions.shape != probabilities.shape:
        raise ValueError('aligned vector metadata required')
    if predictions.dtype != 'float64' or probabilities.dtype != 'float64':
        raise ValueError('float64 metadata required')
    for value in (probabilities, threshold):
        if value.dim is not None and not value.dim.is_dimensionless:
            raise ValueError('probabilities and threshold must be dimensionless')


def witness_estimate_conditional_offsets(observed: AbstractArray, calibration_predictions: AbstractArray,
        calibration_probabilities: AbstractArray, threshold: AbstractScalar) -> AbstractArray:
    _metadata(calibration_predictions, calibration_probabilities, threshold)
    observed.assert_shape_compatible(calibration_predictions)
    if observed.dtype != 'float64':
        raise ValueError('float64 observations required')
    if observed.dim is not None and calibration_predictions.dim is not None and not observed.dim.is_compatible(calibration_predictions.dim):
        raise ValueError('observations and predictions must have compatible units')
    return AbstractArray(shape=(2,), dtype='float64', dim=observed.dim or calibration_predictions.dim)


@register_atom(witness_estimate_conditional_offsets)
def estimate_conditional_offsets(observed: NDArray[np.float64], calibration_predictions: NDArray[np.float64],
        calibration_probabilities: NDArray[np.float64], threshold: float) -> NDArray[np.float64]:
    """Return [median(y-p | q>t), median(p-y | q<t)] in target units.

    Equality contributes to neither group. Both groups must be populated.
    Corrections are signed; the classifier's direction can be wrong. These
    conditional medians do not prove improved accuracy or calibrated q values.
    Inputs are not mutated; all residuals and outputs must remain finite.
    """
    target, prediction = _vector(observed), _vector(calibration_predictions)
    if target.shape != prediction.shape:
        raise ValueError('aligned observations and predictions required')
    probability = _probabilities(calibration_probabilities, target.shape, threshold)
    high, low = probability > threshold, probability < threshold
    if not high.any() or not low.any():
        raise ValueError('both conditional calibration groups must be nonempty')
    with np.errstate(over='ignore', invalid='ignore'):
        residual = target - prediction
        offsets = np.array([np.median(residual[high]), np.median(-residual[low])])
    if not np.isfinite(residual).all() or not np.isfinite(offsets).all():
        raise ValueError('nonfinite residual or conditional median')
    return offsets


def witness_apply_conditional_offsets(predictions: AbstractArray, probabilities: AbstractArray,
        offsets: AbstractArray, threshold: AbstractScalar) -> AbstractArray:
    _metadata(predictions, probabilities, threshold)
    if offsets.shape != (2,) or offsets.dtype != 'float64':
        raise ValueError('two float64 signed offsets required')
    if predictions.dim is not None and offsets.dim is not None and not predictions.dim.is_compatible(offsets.dim):
        raise ValueError('predictions and corrections must have compatible units')
    return AbstractArray(shape=predictions.shape, dtype='float64', dim=predictions.dim or offsets.dim)


@register_atom(witness_apply_conditional_offsets)
def apply_conditional_offsets(predictions: NDArray[np.float64], probabilities: NDArray[np.float64],
        offsets: NDArray[np.float64], threshold: float) -> NDArray[np.float64]:
    """Add offsets[0] for q>t; subtract offsets[1] for q<t; retain ties.

    Values and offsets share target units. No clipping, quantization, unit
    conversion, classifier fitting or domain-specific feature assumptions.
    """
    prediction = _vector(predictions)
    probability = _probabilities(probabilities, prediction.shape, threshold)
    correction = _vector(offsets)
    if correction.shape != (2,):
        raise ValueError('two signed offsets required')
    with np.errstate(over='ignore', invalid='ignore'):
        result = np.where(probability > threshold, prediction + correction[0], prediction)
        result = np.where(probability < threshold, result - correction[1], result)
    if not np.isfinite(result).all():
        raise ValueError('corrected values outside finite float64 range')
    return result
