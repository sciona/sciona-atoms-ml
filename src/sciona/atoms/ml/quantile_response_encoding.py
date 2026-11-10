"""Reusable quantile response encoding with portable state and explicit units.

The caller chooses a normalized numeric response coordinate, training
population and probability interpretation. Registration is not publication.
"""
import numpy as np
from sciona.ghost.abstract import AbstractArray
from sciona.ghost.dimensions import DimensionalSignature
from sciona.ghost.registry import register_atom


def _coordinate(value):
    if type(value) is not str or not value.strip():
        raise ValueError('Explicit normalized response-coordinate identifier required')


def _bins(value):
    if type(value) is not int or value < 1:
        raise ValueError('Positive integer bin count required')


def _policy(soft, precision):
    if type(soft) is not bool:
        raise ValueError('Explicit boolean soft mode required')
    if type(precision) not in (float, int) or not np.isfinite(precision) or precision <= 0:
        raise ValueError('Positive finite precision in inverse squared response units required')


def _state(state, coordinate):
    from sciona.quantile_target_encoding import quantile_midpoints
    _coordinate(coordinate)
    if (type(state) is not dict or set(state) != {'format', 'response_coordinate', 'boundaries'}
            or state['format'] != 'sciona.quantile_response.v1'
            or state['response_coordinate'] != coordinate or type(state['boundaries']) is not list
            or any(type(v) not in (int, float) for v in state['boundaries'])):
        raise ValueError('Matching numeric quantile state required')
    edges = np.asarray(state['boundaries'], dtype=np.float64)
    quantile_midpoints(edges)
    return edges


def _abstract(values, coordinate, rank):
    _coordinate(coordinate)
    if len(values.shape) != rank or values.shape[0] < 1 or values.dtype not in ('float32', 'float64'):
        raise ValueError('Nonempty floating array metadata required')
    if values.dim is not None and not values.dim.is_dimensionless:
        raise ValueError('Normalize response coordinates before quantile encoding')


def _abstract_bins(state, coordinate):
    if (type(state) is not dict or set(state) != {'format', 'response_coordinate', 'n_bins'}
            or state['format'] != 'sciona.quantile_response.abstract.v1'
            or state['response_coordinate'] != coordinate):
        raise ValueError('Matching abstract quantile state required')
    _bins(state['n_bins'])
    return state['n_bins']


def _values(values, rank):
    values = np.asarray(values)
    if values.ndim != rank or values.dtype not in (np.float32, np.float64):
        raise ValueError('Float32 or float64 array required')
    return values


def witness_fit_quantile_response(targets: AbstractArray, response_coordinate: str, n_bins: int) -> dict:
    _abstract(targets, response_coordinate, 1)
    _bins(n_bins)
    return dict(format='sciona.quantile_response.abstract.v1',
                response_coordinate=response_coordinate, n_bins=n_bins)


@register_atom(witness_fit_quantile_response)
def fit_quantile_response(targets: np.ndarray, response_coordinate: str, n_bins: int) -> dict:
    """Fit linearly interpolated training quantiles in a declared numeric coordinate.

    Finite nonempty float32/float64 targets are promoted to float64. Repeated
    boundaries are retained. Caller owns training-only fitting and response
    normalization. Returns portable JSON containing the fitted boundaries.
    """
    from sciona.quantile_target_encoding import quantile_boundaries
    _coordinate(response_coordinate)
    edges = quantile_boundaries(_values(targets, 1), n_bins)
    state = dict(format='sciona.quantile_response.v1', response_coordinate=response_coordinate,
                 boundaries=edges.tolist())
    _state(state, response_coordinate)
    return state


def witness_encode_quantile_response(targets: AbstractArray, response_coordinate: str,
        state: dict, soft: bool, precision: float) -> AbstractArray:
    _abstract(targets, response_coordinate, 1)
    count = _abstract_bins(state, response_coordinate)
    _policy(soft, precision)
    return AbstractArray(shape=(targets.shape[0], count), dtype='float64' if soft else 'float32',
                         dim=DimensionalSignature())


@register_atom(witness_encode_quantile_response)
def encode_quantile_response(targets: np.ndarray, response_coordinate: str,
        state: dict, soft: bool, precision: float) -> np.ndarray:
    """Encode using frozen quantiles; preserve source hard-boundary behavior.

    Soft float64 weights use exp(-precision * squared distance / 2), normalized
    by their direct sum. Underflow/overflow is rejected. Precision has inverse
    squared coordinate units and must be rescaled when response scale changes.
    Hard float32 indicators use left-open/right-closed intervals; minimum and
    out-of-range targets can yield all-zero rows. Positive precision is required
    in both modes but used only in soft mode. Empty populations are rejected.
    """
    from sciona.quantile_target_encoding import encode_quantile_targets
    _policy(soft, precision)
    return encode_quantile_targets(_values(targets, 1), _state(state, response_coordinate),
                                   soft=soft, precision=precision)


def witness_decode_quantile_response(probabilities: AbstractArray, response_coordinate: str,
        state: dict) -> AbstractArray:
    _abstract(probabilities, response_coordinate, 2)
    if probabilities.shape[1] != _abstract_bins(state, response_coordinate):
        raise ValueError('Probability width must match fitted bin count')
    return AbstractArray(shape=(probabilities.shape[0],), dtype='float64', dim=DimensionalSignature())


@register_atom(witness_decode_quantile_response)
def decode_quantile_response(probabilities: np.ndarray, response_coordinate: str, state: dict) -> np.ndarray:
    """Decode normalized probabilities as the fitted-bin midpoint expectation.

    Requires finite nonnegative nonempty float32/float64 rows summing to one
    within rtol=1e-6, atol=1e-8, and exact bin order. Returns finite float64 in
    the declared coordinate; no inverse response transform or calibration is
    inferred. Hard encoding's all-zero rows cannot be decoded as probabilities.
    """
    from sciona.quantile_target_encoding import decode_quantile_probabilities
    with np.errstate(over='ignore', invalid='ignore'):
        result = decode_quantile_probabilities(_values(probabilities, 2), _state(state, response_coordinate))
    if not np.isfinite(result).all():
        raise ValueError('Decoded response overflow')
    return result
