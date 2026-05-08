"""Kernel approximation atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from scipy.fft import fft, ifft
from scipy.linalg import svd
from numpy.typing import NDArray
from sklearn.metrics.pairwise import KERNEL_PARAMS, PAIRWISE_KERNEL_FUNCTIONS, pairwise_kernels

from sciona.ghost.registry import register_atom

from .state_models import NystroemState, PolynomialCountSketchState, RBFSamplerState, SkewedChi2SamplerState
from .witnesses import (
    witness_nystroem_fit,
    witness_nystroem_transform,
    witness_additive_chi2_sampler_transform,
    witness_polynomial_count_sketch_fit,
    witness_polynomial_count_sketch_transform,
    witness_rbf_sampler_fit,
    witness_rbf_sampler_transform,
    witness_skewed_chi2_sampler_fit,
    witness_skewed_chi2_sampler_transform,
)

_SUPPORTED_NYSTROEM_KERNELS = frozenset(PAIRWISE_KERNEL_FUNCTIONS)
_NONNEGATIVE_NYSTROEM_KERNELS = {"additive_chi2", "chi2"}

def _matrix_2d(X: NDArray[np.float64]) -> bool:
    return bool(np.asarray(X).ndim == 2)

def _gamma_valid(gamma: float | str) -> bool:
    return bool(gamma == "scale" or (isinstance(gamma, (int, float)) and not isinstance(gamma, bool) and gamma > 0.0))

def _n_components_valid(n_components: int) -> bool:
    return bool(isinstance(n_components, int) and not isinstance(n_components, bool) and n_components >= 1)

def _random_state_valid(random_state: int | None) -> bool:
    return bool(random_state is None or (isinstance(random_state, int) and not isinstance(random_state, bool)))

def _finite_real(value: float) -> bool:
    return bool(isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(float(value)))

def _nonnegative_real(value: float) -> bool:
    return bool(_finite_real(value) and float(value) >= 0.0)

def _degree_valid(degree: int) -> bool:
    return bool(isinstance(degree, int) and not isinstance(degree, bool) and degree >= 1)

def _nystroem_degree_valid(degree: float | None) -> bool:
    return bool(degree is None or (_finite_real(degree) and float(degree) >= 1.0))

def _nystroem_kernel_valid(kernel: str) -> bool:
    return bool(isinstance(kernel, str) and kernel in _SUPPORTED_NYSTROEM_KERNELS)

def _kernel_params_valid(kernel_params: dict[str, float] | None) -> bool:
    if kernel_params is None:
        return True
    return bool(
        isinstance(kernel_params, dict)
        and all(isinstance(key, str) and _finite_real(value) for key, value in kernel_params.items())
    )

def _n_jobs_valid(n_jobs: int | None) -> bool:
    return bool(n_jobs is None or (isinstance(n_jobs, int) and not isinstance(n_jobs, bool)))

def _positive_values(X: NDArray[np.float64]) -> bool:
    values = np.asarray(X)
    return bool(values.ndim == 2 and np.all(values >= 0.0))

def _skewed_values_valid(X: NDArray[np.float64], state: SkewedChi2SamplerState) -> bool:
    values = np.asarray(X)
    return bool(values.ndim == 2 and np.all(values > -state.skewedness))

def _sample_steps_valid(sample_steps: int) -> bool:
    return bool(isinstance(sample_steps, int) and not isinstance(sample_steps, bool) and sample_steps >= 1)

def _sample_interval_valid(sample_steps: int, sample_interval: float | None) -> bool:
    return bool(
        (sample_interval is None and sample_steps in {1, 2, 3})
        or (
            sample_interval is not None
            and isinstance(sample_interval, (int, float))
            and not isinstance(sample_interval, bool)
            and float(sample_interval) > 0.0
        )
    )

def _rbf_state_valid(state: RBFSamplerState) -> bool:
    return bool(
        state.random_weights.shape == (state.n_features_in, state.n_components)
        and state.random_offset.shape == (state.n_components,)
        and state.gamma > 0.0
        and state.n_components >= 1
        and state.n_features_in >= 1
        and _random_state_valid(state.random_state)
        and np.all(np.isfinite(state.random_weights))
        and np.all(np.isfinite(state.random_offset))
    )

def _skewed_state_valid(state: SkewedChi2SamplerState) -> bool:
    return bool(
        state.random_weights.shape == (state.n_features_in, state.n_components)
        and state.random_offset.shape == (state.n_components,)
        and np.isfinite(state.skewedness)
        and state.n_components >= 1
        and state.n_features_in >= 1
        and _random_state_valid(state.random_state)
        and np.all(np.isfinite(state.random_weights))
        and np.all(np.isfinite(state.random_offset))
    )

def _polynomial_state_valid(state: PolynomialCountSketchState) -> bool:
    hashed_features = state.n_features_in + (1 if state.coef0 != 0.0 else 0)
    return bool(
        state.index_hash.shape == (state.degree, hashed_features)
        and state.bit_hash.shape == (state.degree, hashed_features)
        and state.index_hash.dtype == np.int64
        and state.bit_hash.dtype == np.int64
        and state.gamma > 0.0
        and state.degree >= 1
        and state.coef0 >= 0.0
        and state.n_components >= 1
        and state.n_features_in >= 1
        and _random_state_valid(state.random_state)
        and np.all((state.index_hash >= 0) & (state.index_hash < state.n_components))
        and np.all(np.isin(state.bit_hash, [-1, 1]))
    )

def _nystroem_state_valid(state: NystroemState) -> bool:
    return bool(
        state.components.shape == (state.n_components, state.n_features_in)
        and state.component_indices.shape == (state.n_components,)
        and state.normalization.shape == (state.n_components, state.n_components)
        and state.components.dtype == np.float64
        and state.component_indices.dtype == np.int64
        and state.normalization.dtype == np.float64
        and _nystroem_kernel_valid(state.kernel)
        and _kernel_params_valid(state.kernel_params)
        and state.n_components >= 1
        and state.n_features_in >= 1
        and _random_state_valid(state.random_state)
        and _n_jobs_valid(state.n_jobs)
        and np.all(state.component_indices >= 0)
        and np.all(np.isfinite(state.components))
        and np.all(np.isfinite(state.normalization))
    )

def _feature_count_matches(X: NDArray[np.float64], state: RBFSamplerState) -> bool:
    return bool(np.asarray(X).ndim == 2 and np.asarray(X).shape[1] == state.n_features_in)

def _transformed_valid(result: NDArray[np.float64], X: NDArray[np.float64], state: RBFSamplerState) -> bool:
    values = np.asarray(result)
    return bool(values.shape == (np.asarray(X).shape[0], state.n_components) and np.all(np.isfinite(values)))

def _skewed_feature_count_matches(X: NDArray[np.float64], state: SkewedChi2SamplerState) -> bool:
    return bool(np.asarray(X).ndim == 2 and np.asarray(X).shape[1] == state.n_features_in)

def _skewed_transformed_valid(result: NDArray[np.float64], X: NDArray[np.float64], state: SkewedChi2SamplerState) -> bool:
    values = np.asarray(result)
    return bool(values.shape == (np.asarray(X).shape[0], state.n_components) and np.all(np.isfinite(values)))

def _polynomial_feature_count_matches(X: NDArray[np.float64], state: PolynomialCountSketchState) -> bool:
    return bool(np.asarray(X).ndim == 2 and np.asarray(X).shape[1] == state.n_features_in)

def _polynomial_transformed_valid(result: NDArray[np.float64], X: NDArray[np.float64], state: PolynomialCountSketchState) -> bool:
    values = np.asarray(result)
    return bool(values.shape == (np.asarray(X).shape[0], state.n_components) and np.all(np.isfinite(values)))

def _nystroem_feature_count_matches(X: NDArray[np.float64], state: NystroemState) -> bool:
    return bool(np.asarray(X).ndim == 2 and np.asarray(X).shape[1] == state.n_features_in)

def _nystroem_values_valid(X: NDArray[np.float64], kernel: str) -> bool:
    values = np.asarray(X)
    return bool(values.ndim == 2 and (kernel not in _NONNEGATIVE_NYSTROEM_KERNELS or np.all(values >= 0.0)))

def _nystroem_state_values_valid(X: NDArray[np.float64], state: NystroemState) -> bool:
    return _nystroem_values_valid(X, state.kernel)

def _nystroem_transformed_valid(result: NDArray[np.float64], X: NDArray[np.float64], state: NystroemState) -> bool:
    values = np.asarray(result)
    return bool(values.shape == (np.asarray(X).shape[0], state.n_components) and np.all(np.isfinite(values)))

def _additive_transformed_valid(result: NDArray[np.float64], X: NDArray[np.float64], sample_steps: int) -> bool:
    values = np.asarray(result)
    expected_shape = (np.asarray(X).shape[0], np.asarray(X).shape[1] * (2 * sample_steps - 1))
    return bool(values.shape == expected_shape and np.all(np.isfinite(values)))

def _resolve_sample_interval(sample_steps: int, sample_interval: float | None) -> float:
    if sample_interval is not None:
        return float(sample_interval)
    if sample_steps == 1:
        return 0.8
    if sample_steps == 2:
        return 0.5
    if sample_steps == 3:
        return 0.4
    raise ValueError("sample_interval is required for sample_steps outside {1, 2, 3}")

def _resolve_nystroem_kernel_params(
    kernel: str,
    gamma: float | None,
    coef0: float | None,
    degree: float | None,
    kernel_params: dict[str, float] | None,
) -> dict[str, float]:
    params = dict(kernel_params or {})
    provided = {"gamma": gamma, "coef0": coef0, "degree": degree}
    for param in KERNEL_PARAMS[kernel]:
        value = provided[param]
        if value is not None:
            params[param] = float(value)
    return params

@register_atom(witness_rbf_sampler_fit)
@icontract.require(lambda X: _matrix_2d(X), "X must be 2D")
@icontract.require(lambda gamma: _gamma_valid(gamma), "gamma must be positive or 'scale'")
@icontract.require(lambda n_components: _n_components_valid(n_components), "n_components must be positive")
@icontract.require(lambda random_state: _random_state_valid(random_state), "random_state must be None or an integer seed")
@icontract.ensure(lambda result: _rbf_state_valid(result), "RBF sampler state must contain finite random Fourier weights")
def rbf_sampler_fit(
    X: NDArray[np.float64],
    *,
    gamma: float | str = 1.0,
    n_components: int = 100,
    random_state: int | None = None,
) -> RBFSamplerState:
    from sklearn.utils import check_array, check_random_state
    """Fit random Fourier weights for a dense RBF kernel approximation."""
    checked_x = check_array(X, dtype=np.float64, ensure_2d=True)
    rng = check_random_state(random_state)
    n_features = checked_x.shape[1]
    if gamma == "scale":
        variance = float(np.var(checked_x))
        gamma_value = 1.0 / (n_features * variance) if variance != 0.0 else 1.0
    else:
        gamma_value = float(gamma)
    random_weights = np.sqrt(2.0 * gamma_value) * rng.normal(size=(n_features, n_components))
    random_offset = rng.uniform(0.0, 2.0 * np.pi, size=n_components)
    return RBFSamplerState(
        random_weights=np.asarray(random_weights, dtype=np.float64),
        random_offset=np.asarray(random_offset, dtype=np.float64),
        gamma=gamma_value,
        n_components=int(n_components),
        n_features_in=int(n_features),
        random_state=random_state,
    )

@register_atom(witness_rbf_sampler_transform)
@icontract.require(lambda X: _matrix_2d(X), "X must be 2D")
@icontract.require(lambda X, state: _feature_count_matches(X, state), "X feature count must match fitted RBF sampler state")
@icontract.require(lambda state: _rbf_state_valid(state), "state must be a fitted RBF sampler")
@icontract.ensure(lambda result, X, state: _transformed_valid(result, X, state), "random Fourier features must have the fitted component width")
def rbf_sampler_transform(
    X: NDArray[np.float64],
    state: RBFSamplerState,
) -> NDArray[np.float64]:
    from sklearn.utils import check_array, check_random_state
    from sklearn.utils.extmath import safe_sparse_dot
    """Project samples into fitted RBF random Fourier features."""
    checked_x = check_array(X, dtype=np.float64, ensure_2d=True)
    projection = safe_sparse_dot(checked_x, state.random_weights)
    projection += state.random_offset
    np.cos(projection, projection)
    projection *= np.sqrt(2.0 / state.n_components)
    return np.asarray(projection, dtype=np.float64)

@register_atom(witness_skewed_chi2_sampler_fit)
@icontract.require(lambda X: _matrix_2d(X), "X must be 2D")
@icontract.require(lambda skewedness: _finite_real(skewedness), "skewedness must be finite")
@icontract.require(lambda n_components: _n_components_valid(n_components), "n_components must be positive")
@icontract.require(lambda random_state: _random_state_valid(random_state), "random_state must be None or an integer seed")
@icontract.ensure(lambda result: _skewed_state_valid(result), "skewed chi-square state must contain finite random Fourier weights")
def skewed_chi2_sampler_fit(
    X: NDArray[np.float64],
    *,
    skewedness: float = 1.0,
    n_components: int = 100,
    random_state: int | None = None,
) -> SkewedChi2SamplerState:
    from sklearn.utils import check_array, check_random_state
    """Fit random Fourier weights for a dense skewed chi-square approximation."""
    checked_x = check_array(X, dtype=np.float64, ensure_2d=True)
    rng = check_random_state(random_state)
    n_features = checked_x.shape[1]
    uniform = rng.uniform(size=(n_features, n_components))
    random_weights = (1.0 / np.pi) * np.log(np.tan((np.pi / 2.0) * uniform))
    random_offset = rng.uniform(0.0, 2.0 * np.pi, size=n_components)
    return SkewedChi2SamplerState(
        random_weights=np.asarray(random_weights, dtype=np.float64),
        random_offset=np.asarray(random_offset, dtype=np.float64),
        skewedness=float(skewedness),
        n_components=int(n_components),
        n_features_in=int(n_features),
        random_state=random_state,
    )

@register_atom(witness_skewed_chi2_sampler_transform)
@icontract.require(lambda X: _matrix_2d(X), "X must be 2D")
@icontract.require(lambda X, state: _skewed_feature_count_matches(X, state), "X feature count must match fitted skewed chi-square state")
@icontract.require(lambda X, state: _skewed_values_valid(X, state), "X entries must be greater than -skewedness")
@icontract.require(lambda state: _skewed_state_valid(state), "state must be a fitted skewed chi-square sampler")
@icontract.ensure(lambda result, X, state: _skewed_transformed_valid(result, X, state), "skewed chi-square features must have the fitted component width")
def skewed_chi2_sampler_transform(
    X: NDArray[np.float64],
    state: SkewedChi2SamplerState,
) -> NDArray[np.float64]:
    from sklearn.utils import check_array, check_random_state
    from sklearn.utils.extmath import safe_sparse_dot
    """Project samples into fitted skewed chi-square random Fourier features."""
    checked_x = check_array(X, dtype=np.float64, ensure_2d=True, copy=True)
    checked_x += state.skewedness
    np.log(checked_x, checked_x)
    projection = safe_sparse_dot(checked_x, state.random_weights)
    projection += state.random_offset
    np.cos(projection, projection)
    projection *= np.sqrt(2.0 / state.n_components)
    return np.asarray(projection, dtype=np.float64)

@register_atom(witness_additive_chi2_sampler_transform)
@icontract.require(lambda X: _matrix_2d(X), "X must be 2D")
@icontract.require(lambda X: _positive_values(X), "X must be non-negative")
@icontract.require(lambda sample_steps: _sample_steps_valid(sample_steps), "sample_steps must be positive")
@icontract.require(lambda sample_steps, sample_interval: _sample_interval_valid(sample_steps, sample_interval), "sample_interval must be positive or omitted for sample_steps in {1, 2, 3}")
@icontract.ensure(lambda result, X, sample_steps: _additive_transformed_valid(result, X, sample_steps), "additive chi-square features must match the expanded width")
def additive_chi2_sampler_transform(
    X: NDArray[np.float64],
    *,
    sample_steps: int = 2,
    sample_interval: float | None = None,
) -> NDArray[np.float64]:
    from sklearn.utils import check_array, check_random_state
    """Compute dense explicit additive chi-square kernel features."""
    checked_x = check_array(X, dtype=np.float64, ensure_2d=True)
    interval = _resolve_sample_interval(sample_steps, sample_interval)
    non_zero = checked_x != 0.0
    non_zero_values = checked_x[non_zero]

    first_step = np.zeros_like(checked_x)
    first_step[non_zero] = np.sqrt(non_zero_values * interval)
    transformed = [first_step]

    log_step = interval * np.log(non_zero_values)
    scaled_step = 2.0 * non_zero_values * interval
    for step_index in range(1, sample_steps):
        factor = np.sqrt(scaled_step / np.cosh(np.pi * step_index * interval))
        cosine_step = np.zeros_like(checked_x)
        cosine_step[non_zero] = factor * np.cos(step_index * log_step)
        transformed.append(cosine_step)

        sine_step = np.zeros_like(checked_x)
        sine_step[non_zero] = factor * np.sin(step_index * log_step)
        transformed.append(sine_step)

    return np.asarray(np.hstack(transformed), dtype=np.float64)

@register_atom(witness_polynomial_count_sketch_fit)
@icontract.require(lambda X: _matrix_2d(X), "X must be 2D")
@icontract.require(lambda gamma: _finite_real(gamma) and gamma > 0.0, "gamma must be positive")
@icontract.require(lambda degree: _degree_valid(degree), "degree must be positive")
@icontract.require(lambda coef0: _nonnegative_real(coef0), "coef0 must be non-negative")
@icontract.require(lambda n_components: _n_components_valid(n_components), "n_components must be positive")
@icontract.require(lambda random_state: _random_state_valid(random_state), "random_state must be None or an integer seed")
@icontract.ensure(lambda result: _polynomial_state_valid(result), "polynomial count-sketch state must contain valid hash tables")
def polynomial_count_sketch_fit(
    X: NDArray[np.float64],
    *,
    gamma: float = 1.0,
    degree: int = 2,
    coef0: float = 0.0,
    n_components: int = 100,
    random_state: int | None = None,
) -> PolynomialCountSketchState:
    from sklearn.utils import check_array, check_random_state
    """Fit Tensor Sketch hash tables for dense polynomial kernel features."""
    checked_x = check_array(X, dtype=np.float64, ensure_2d=True)
    rng = check_random_state(random_state)
    hashed_features = checked_x.shape[1] + (1 if coef0 != 0.0 else 0)
    index_hash = rng.randint(0, high=n_components, size=(degree, hashed_features))
    bit_hash = rng.choice(a=[-1, 1], size=(degree, hashed_features))
    return PolynomialCountSketchState(
        index_hash=np.asarray(index_hash, dtype=np.int64),
        bit_hash=np.asarray(bit_hash, dtype=np.int64),
        gamma=float(gamma),
        degree=int(degree),
        coef0=float(coef0),
        n_components=int(n_components),
        n_features_in=int(checked_x.shape[1]),
        random_state=random_state,
    )

@register_atom(witness_polynomial_count_sketch_transform)
@icontract.require(lambda X: _matrix_2d(X), "X must be 2D")
@icontract.require(lambda X, state: _polynomial_feature_count_matches(X, state), "X feature count must match fitted polynomial sketch state")
@icontract.require(lambda state: _polynomial_state_valid(state), "state must be a fitted polynomial count sketch")
@icontract.ensure(lambda result, X, state: _polynomial_transformed_valid(result, X, state), "polynomial count-sketch features must have the fitted component width")
def polynomial_count_sketch_transform(
    X: NDArray[np.float64],
    state: PolynomialCountSketchState,
) -> NDArray[np.float64]:
    from sklearn.utils import check_array, check_random_state
    """Project samples into fitted polynomial Tensor Sketch features."""
    checked_x = check_array(X, dtype=np.float64, ensure_2d=True)
    scaled_x = np.sqrt(state.gamma) * checked_x
    if state.coef0 != 0.0:
        scaled_x = np.hstack([scaled_x, np.sqrt(state.coef0) * np.ones((scaled_x.shape[0], 1))])

    count_sketches = np.zeros((scaled_x.shape[0], state.degree, state.n_components), dtype=np.float64)
    for feature_index in range(scaled_x.shape[1]):
        for degree_index in range(state.degree):
            hash_index = state.index_hash[degree_index, feature_index]
            hash_bit = state.bit_hash[degree_index, feature_index]
            count_sketches[:, degree_index, hash_index] += hash_bit * scaled_x[:, feature_index]

    sketches_fft = fft(count_sketches, axis=2, overwrite_x=True)
    product_fft = np.prod(sketches_fft, axis=1)
    data_sketch = np.real(ifft(product_fft, overwrite_x=True))
    return np.asarray(data_sketch, dtype=np.float64)

@register_atom(witness_nystroem_fit)
@icontract.require(lambda X: _matrix_2d(X), "X must be 2D")
@icontract.require(lambda kernel: _nystroem_kernel_valid(kernel), "kernel must be a built-in pairwise kernel")
@icontract.require(lambda gamma: gamma is None or (_finite_real(gamma) and gamma >= 0.0), "gamma must be non-negative or None")
@icontract.require(lambda coef0: coef0 is None or _finite_real(coef0), "coef0 must be finite or None")
@icontract.require(lambda degree: _nystroem_degree_valid(degree), "degree must be at least one or None")
@icontract.require(lambda kernel_params: _kernel_params_valid(kernel_params), "kernel_params must contain finite numeric values")
@icontract.require(lambda n_components: _n_components_valid(n_components), "n_components must be positive")
@icontract.require(lambda random_state: _random_state_valid(random_state), "random_state must be None or an integer seed")
@icontract.require(lambda n_jobs: _n_jobs_valid(n_jobs), "n_jobs must be None or an integer")
@icontract.require(lambda X, kernel: _nystroem_values_valid(X, kernel), "chi-square kernels require non-negative X")
@icontract.ensure(lambda result: _nystroem_state_valid(result), "Nystroem state must contain finite basis and normalization arrays")
def nystroem_fit(
    X: NDArray[np.float64],
    *,
    kernel: str = "rbf",
    gamma: float | None = None,
    coef0: float | None = None,
    degree: float | None = None,
    kernel_params: dict[str, float] | None = None,
    n_components: int = 100,
    random_state: int | None = None,
    n_jobs: int | None = None,
) -> NystroemState:
    from sklearn.utils import check_array, check_random_state
    """Fit dense Nystroem basis components and normalization matrix."""
    checked_x = check_array(X, dtype=np.float64, ensure_2d=True)
    rng = check_random_state(random_state)
    actual_components = min(checked_x.shape[0], int(n_components))
    basis_indices = rng.permutation(checked_x.shape[0])[:actual_components]
    basis = checked_x[basis_indices]
    resolved_params = _resolve_nystroem_kernel_params(kernel, gamma, coef0, degree, kernel_params)
    basis_kernel = pairwise_kernels(
        basis,
        metric=kernel,
        filter_params=True,
        n_jobs=n_jobs,
        **resolved_params,
    )
    left, singular_values, right = svd(basis_kernel)
    singular_values = np.maximum(singular_values, 1e-12)
    normalization = np.dot(left / np.sqrt(singular_values), right)
    return NystroemState(
        components=np.asarray(basis, dtype=np.float64),
        component_indices=np.asarray(basis_indices, dtype=np.int64),
        normalization=np.asarray(normalization, dtype=np.float64),
        kernel=kernel,
        kernel_params=resolved_params,
        n_components=int(actual_components),
        n_features_in=int(checked_x.shape[1]),
        random_state=random_state,
        n_jobs=n_jobs,
    )

@register_atom(witness_nystroem_transform)
@icontract.require(lambda X: _matrix_2d(X), "X must be 2D")
@icontract.require(lambda X, state: _nystroem_feature_count_matches(X, state), "X feature count must match fitted Nystroem state")
@icontract.require(lambda X, state: _nystroem_state_values_valid(X, state), "chi-square kernels require non-negative X")
@icontract.require(lambda state: _nystroem_state_valid(state), "state must be a fitted Nystroem state")
@icontract.ensure(lambda result, X, state: _nystroem_transformed_valid(result, X, state), "Nystroem features must have the fitted component width")
def nystroem_transform(
    X: NDArray[np.float64],
    state: NystroemState,
) -> NDArray[np.float64]:
    from sklearn.utils import check_array, check_random_state
    """Project samples with a fitted dense Nystroem kernel map."""
    checked_x = check_array(X, dtype=np.float64, ensure_2d=True)
    embedded = pairwise_kernels(
        checked_x,
        state.components,
        metric=state.kernel,
        filter_params=True,
        n_jobs=state.n_jobs,
        **state.kernel_params,
    )
    return np.asarray(np.dot(embedded, state.normalization.T), dtype=np.float64)
