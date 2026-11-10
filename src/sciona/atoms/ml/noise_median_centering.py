"""Explicit additive perturbation followed by independent window centering.

This is an independently implemented numeric operation. Registration is not
catalog approval and does not qualify a complete application graph.
"""
import numpy as np
from sciona.ghost.abstract import AbstractArray
from sciona.ghost.registry import register_atom


def _contract(shape, noise_shape, dtype, noise_dtype, amplitude_coordinate):
    if len(shape) != 2 or min(shape) < 1 or shape != noise_shape:
        raise ValueError('Matching nonempty windows-by-samples arrays required')
    if np.dtype(dtype).kind not in 'iuf' or np.dtype(noise_dtype).kind not in 'iuf':
        raise ValueError('Real numeric signal and additive noise required')
    if type(amplitude_coordinate) is not str or not amplitude_coordinate.strip():
        raise ValueError('Explicit common amplitude coordinate required')


def witness_add_noise_median_center(windows: AbstractArray, additive_noise: AbstractArray,
                                    amplitude_coordinate: str) -> AbstractArray:
    _contract(windows.shape, additive_noise.shape, windows.dtype, additive_noise.dtype,
              amplitude_coordinate)
    if windows.dim != additive_noise.dim:
        raise ValueError('Matching amplitude dimensions required')
    return AbstractArray(shape=windows.shape, dtype='float64', dim=windows.dim)


@register_atom(witness_add_noise_median_center)
def add_noise_median_center(windows: np.ndarray, additive_noise: np.ndarray,
                           amplitude_coordinate: str) -> np.ndarray:
    """Add caller-supplied noise, then subtract each row's perturbed median.

    Inputs are matching nonempty real arrays with axes (windows, samples), in
    the same explicitly named amplitude coordinate and scale. No broadcasting,
    unit conversion, windowing, sampling-rate inference or random generation
    occurs. Caller owns noise distribution, amplitude, seed and alignment;
    Gaussian noise is an application choice, not a universal denoising rule.
    Both inputs are converted to float64 before addition. Inputs and outputs
    must be finite; overflow rejects. Integer values beyond float64 precision
    may round. Returns independent float64 storage without changing inputs.
    Window length and overlap are caller policy; splitting a window generally
    changes its centered values. Median centering loses the absolute baseline
    and adds perturbations, so this operation is neither lossless nor guaranteed
    to improve downstream predictions. Executes serially without workers.
    """
    if not isinstance(windows, np.ndarray) or not isinstance(additive_noise, np.ndarray):
        raise ValueError('NumPy arrays required')
    _contract(windows.shape, additive_noise.shape, windows.dtype, additive_noise.dtype,
              amplitude_coordinate)
    with np.errstate(over='ignore', invalid='ignore'):
        values = windows.astype(np.float64, copy=True)
        noise = additive_noise.astype(np.float64, copy=False)
        if not np.isfinite(values).all() or not np.isfinite(noise).all():
            raise ValueError('Finite float64-representable inputs required')
        values += noise
        if not np.isfinite(values).all():
            raise ValueError('Additive overflow')
        values -= np.median(values, axis=1, keepdims=True)
    if not np.isfinite(values).all():
        raise ValueError('Median centering overflow')
    return values
