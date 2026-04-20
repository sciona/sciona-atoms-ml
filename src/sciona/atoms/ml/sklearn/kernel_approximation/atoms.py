"""Kernel approximation atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray
from sklearn.utils import check_array, check_random_state
from sklearn.utils.extmath import safe_sparse_dot

from sciona.ghost.registry import register_atom

from .state_models import RBFSamplerState
from .witnesses import witness_rbf_sampler_fit, witness_rbf_sampler_transform


def _matrix_2d(X: NDArray[np.float64]) -> bool:
    return bool(np.asarray(X).ndim == 2)


def _gamma_valid(gamma: float | str) -> bool:
    return bool(gamma == "scale" or (isinstance(gamma, (int, float)) and not isinstance(gamma, bool) and gamma > 0.0))


def _n_components_valid(n_components: int) -> bool:
    return bool(isinstance(n_components, int) and not isinstance(n_components, bool) and n_components >= 1)


def _random_state_valid(random_state: int | None) -> bool:
    return bool(random_state is None or (isinstance(random_state, int) and not isinstance(random_state, bool)))


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


def _feature_count_matches(X: NDArray[np.float64], state: RBFSamplerState) -> bool:
    return bool(np.asarray(X).ndim == 2 and np.asarray(X).shape[1] == state.n_features_in)


def _transformed_valid(result: NDArray[np.float64], X: NDArray[np.float64], state: RBFSamplerState) -> bool:
    values = np.asarray(result)
    return bool(values.shape == (np.asarray(X).shape[0], state.n_components) and np.all(np.isfinite(values)))


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
