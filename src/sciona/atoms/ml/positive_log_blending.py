"""Reusable positive log-space blending with explicit numeric response policy.

Registration is not catalog publication. Abstract witnesses check shape and
policy compatibility; concrete code enforces numeric and solver boundaries.
"""
import numpy as np
from numpy.typing import NDArray
from threadpoolctl import threadpool_limits

from sciona.ghost.abstract import AbstractArray
from sciona.ghost.dimensions import DimensionalSignature
from sciona.ghost.registry import register_atom


def _identity(columns, coordinate, width):
    if (not isinstance(columns, list) or len(columns) != width
            or any(type(c) is not str or not c for c in columns) or len(set(columns)) != width):
        raise ValueError('Unique ordered model-column identifiers required')
    if type(coordinate) is not str or not coordinate.strip():
        raise ValueError('Explicit numeric response-coordinate identifier required')


def _metadata(predictions, columns, coordinate):
    if len(predictions.shape) != 2 or predictions.shape[1] < 1 or predictions.dtype != 'float64':
        raise ValueError('Float64 prediction-matrix metadata required')
    _identity(columns, coordinate, predictions.shape[1])
    if predictions.dim is not None and not predictions.dim.is_dimensionless:
        raise ValueError('Normalize responses to the declared numeric coordinate before log1p')


def _policy(bounds, controls):
    from sciona._runtime.ml.positive_log_blend import PositiveLogBlend
    PositiveLogBlend((0.,), 0., tuple(bounds))
    if not isinstance(controls, dict) or set(controls) != {'alpha', 'max_iter', 'tolerance', 'seed'}:
        raise ValueError('Explicit complete Lasso solver policy required')
    if any(type(controls[k]) not in (int, float) or not np.isfinite(controls[k]) or controls[k] <= 0
           for k in ('alpha', 'tolerance')):
        raise ValueError('Positive finite alpha and tolerance required')
    if type(controls['max_iter']) is not int or controls['max_iter'] < 1:
        raise ValueError('Positive iteration limit required')
    if type(controls['seed']) is not int or not 0 <= controls['seed'] < 2**32:
        raise ValueError('Valid integer random seed required')


def witness_fit_positive_log_blend(predictions: AbstractArray, targets: AbstractArray,
        model_columns: list, response_coordinate: str, clip_bounds: list, controls: dict) -> dict:
    _metadata(predictions, model_columns, response_coordinate)
    _policy(clip_bounds, controls)
    if predictions.shape[0] < 2 or targets.shape != (predictions.shape[0],) or targets.dtype != 'float64':
        raise ValueError('At least two aligned float64 calibration rows required')
    if targets.dim is not None and not targets.dim.is_dimensionless:
        raise ValueError('Targets must use the declared normalized response coordinate')
    return dict(format='sciona.positive_log_blend.abstract.v1', model_columns=list(model_columns),
                response_coordinate=response_coordinate)


@register_atom(witness_fit_positive_log_blend)
def fit_positive_log_blend(predictions: NDArray[np.float64], targets: NDArray[np.float64],
        model_columns: list, response_coordinate: str, clip_bounds: list, controls: dict) -> dict:
    """Fit positive Lasso on clipped log1p calibration scores and nonnegative targets.

    Caller supplies held-out or out-of-fold scores in a common normalized
    response coordinate and owns population independence. Column IDs identify
    model outputs, not data rows. Bounds and solver controls are explicit.
    Returns portable numeric state; coefficients need not sum to one.
    """
    from sciona._runtime.ml.positive_log_blend import fit_positive_log_blend as fit
    x, y = np.asarray(predictions), np.asarray(targets)
    if x.ndim != 2 or x.dtype != np.float64 or y.dtype != np.float64:
        raise ValueError('Float64 prediction matrix and target vector required')
    _identity(model_columns, response_coordinate, x.shape[1])
    _policy(clip_bounds, controls)
    with threadpool_limits(limits=1):
        fitted = fit(x, y, clip_bounds=clip_bounds, **controls)
    return dict(format='sciona.positive_log_blend.v1', model_columns=list(model_columns),
                response_coordinate=response_coordinate, coefficients=list(fitted.coefficients),
                intercept=fitted.intercept, clip_bounds=list(fitted.clip_bounds))


def witness_apply_positive_log_blend(predictions: AbstractArray, model_columns: list,
        response_coordinate: str, state: dict) -> AbstractArray:
    _metadata(predictions, model_columns, response_coordinate)
    expected = dict(format='sciona.positive_log_blend.abstract.v1', model_columns=model_columns,
                    response_coordinate=response_coordinate)
    if state != expected:
        raise ValueError('Matching abstract blend state required')
    return AbstractArray(shape=(predictions.shape[0],), dtype='float64', dim=DimensionalSignature())


@register_atom(witness_apply_positive_log_blend)
def apply_positive_log_blend(predictions: NDArray[np.float64], model_columns: list,
        response_coordinate: str, state: dict) -> NDArray[np.float64]:
    """Apply portable blend state without fitting; enforce column and coordinate identity.

    Output is in the declared numeric response coordinate. A free intercept can
    yield negative outputs; this atom does not silently add a final clamp.
    Empty query matrices are supported. No executable deserialization occurs.
    """
    from sciona._runtime.ml.positive_log_blend import PositiveLogBlend
    x = np.asarray(predictions)
    if x.ndim != 2 or x.dtype != np.float64:
        raise ValueError('Float64 prediction matrix required')
    _identity(model_columns, response_coordinate, x.shape[1])
    if (not isinstance(state, dict) or set(state) != {'format', 'model_columns', 'response_coordinate',
            'coefficients', 'intercept', 'clip_bounds'} or state['format'] != 'sciona.positive_log_blend.v1'
            or state['model_columns'] != model_columns or state['response_coordinate'] != response_coordinate):
        raise ValueError('Matching numeric blend state required')
    fitted = PositiveLogBlend(state['coefficients'], state['intercept'], state['clip_bounds'])
    with threadpool_limits(limits=1):
        return fitted.predict(x)
