"""Versioned per-example momentum updates from the reviewed public algorithms.

Algorithm sources (Apache-2.0): dongyp13/Non-Targeted-Adversarial-Attacks
attack_iter.py and dongyp13/Targeted-Adversarial-Attack target_attack.py.
These providers preserve the original denominator formulas on their nonzero
domain. They do not change the older globally normalized providers.
"""
import numpy as np
from numpy.typing import NDArray
from sciona.ghost.abstract import AbstractArray
from sciona.ghost.registry import register_atom


def witness_per_example_momentum(gradient: AbstractArray, previous_accumulated: AbstractArray,
                                 momentum: float = 1.0) -> AbstractArray:
    if gradient.shape != previous_accumulated.shape or len(gradient.shape) != 4:
        raise ValueError('Identical nonempty NHWC shapes required')
    return AbstractArray(shape=gradient.shape, dtype='float64')


def _inputs(gradient, previous_accumulated, momentum):
    values = []
    for item in (gradient, previous_accumulated):
        raw = np.asarray(item)
        if raw.dtype.kind not in 'iuf' or raw.ndim != 4 or not raw.size:
            raise ValueError('Finite real nonempty NHWC arrays required')
        with np.errstate(over='raise', invalid='raise'):
            value = raw.astype(np.float64)
        if not np.all(np.isfinite(value)):
            raise ValueError('Finite gradient and momentum state required')
        values.append(value)
    if values[0].shape != values[1].shape:
        raise ValueError('Gradient and momentum state shapes must match')
    raw = np.asarray(momentum)
    if raw.dtype.kind not in 'iuf' or raw.ndim or not np.isfinite(raw) or raw < 0:
        raise ValueError('Finite nonnegative scalar momentum required')
    return *values, float(raw)


def _divide(values, denominator):
    if np.any(~np.isfinite(denominator)) or np.any(denominator <= 0):
        raise ValueError('Each normalization denominator must be finite and positive')
    return values/denominator


@register_atom(witness_per_example_momentum)
def per_example_l1_momentum(gradient: NDArray[np.float64], previous_accumulated: NDArray[np.float64],
                            momentum: float = 1.0) -> NDArray[np.float64]:
    """Normalize each NHWC gradient by its own mean absolute value and add history.

    Return momentum*previous_accumulated + gradient/mean(abs(gradient)), with
    reduction over H,W,C separately for each example. Inputs convert to float64;
    shapes must match and be nonempty. Reject nonfinite/boolean/complex inputs,
    nonfinite intermediates and zero denominators. No stabilizing epsilon is
    added. State is explicit and inputs are not mutated. This is an update
    primitive; model evaluation, autodiff and attack iteration are external.
    """
    g, previous, coefficient = _inputs(gradient, previous_accumulated, momentum)
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        result = coefficient*previous + _divide(g, np.mean(np.abs(g), axis=(1, 2, 3), keepdims=True))
    if not np.all(np.isfinite(result)):
        raise ValueError('Finite accumulated gradient required')
    return result


@register_atom(witness_per_example_momentum)
def per_example_std_momentum(gradient: NDArray[np.float64], previous_accumulated: NDArray[np.float64],
                             momentum: float = 1.0) -> NDArray[np.float64]:
    """Apply both targeted-variant population-std normalizations per example.

    Divide the gradient by its per-example population standard deviation,
    add momentum*previous_accumulated, then divide that result by its own
    per-example population standard deviation. There is no mean subtraction
    from either returned tensor and no denominator epsilon. Both deviations
    must be finite and positive; degenerate examples fail explicitly.
    Float64 conversion, matching nonempty NHWC shapes, finite real inputs and
    nonnegative scalar momentum are required. Inputs are not mutated.
    """
    g, previous, coefficient = _inputs(gradient, previous_accumulated, momentum)
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        normalized = _divide(g, np.std(g, axis=(1, 2, 3), keepdims=True))
        accumulated = coefficient*previous + normalized
        result = _divide(accumulated, np.std(accumulated, axis=(1, 2, 3), keepdims=True))
    if not np.all(np.isfinite(result)):
        raise ValueError('Finite normalized accumulated gradient required')
    return result
