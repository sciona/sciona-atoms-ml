"""Kernel approximation atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray
from sklearn.utils import check_array, check_random_state
from sklearn.utils.extmath import safe_sparse_dot

from sciona.ghost.registry import register_atom

from .state_models import RBFSamplerState, SkewedChi2SamplerState
from .witnesses import (
    witness_additive_chi2_sampler_transform,
    witness_rbf_sampler_fit,
    witness_rbf_sampler_transform,
    witness_skewed_chi2_sampler_fit,
    witness_skewed_chi2_sampler_transform,
)


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
