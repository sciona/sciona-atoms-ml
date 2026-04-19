"""Random projection estimator atoms adapted from scikit-learn."""

from __future__ import annotations

import warnings

import icontract
import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray
from scipy import linalg
from sklearn.exceptions import DataDimensionalityWarning
from sklearn.random_projection import sample_without_replacement
from sklearn.utils import check_array, check_random_state
from sklearn.utils.extmath import safe_sparse_dot

from sciona.ghost.registry import register_atom

from .state_models import MatrixLike, RandomProjectionState
from .witnesses import (
    witness_gaussian_random_projection_fit,
    witness_gaussian_random_projection_transform,
    witness_random_projection_inverse_transform,
    witness_sparse_random_projection_fit,
    witness_sparse_random_projection_transform,
)


def _is_2d(X: MatrixLike) -> bool:
    return bool(getattr(X, "ndim", 0) == 2)


def _feature_count(X: MatrixLike) -> int:
    return int(X.shape[1])


def _row_count(X: MatrixLike) -> int:
    return int(X.shape[0])


def _valid_n_components(n_components: int | str) -> bool:
    return n_components == "auto" or (isinstance(n_components, int) and n_components >= 1)


def _valid_fit_eps(n_components: int | str, eps: float) -> bool:
    if n_components == "auto":
        return 0.0 < eps < 1.0
    return eps > 0.0


def _valid_density(density: float | str) -> bool:
    return density == "auto" or (isinstance(density, (float, int)) and 0.0 < float(density) <= 1.0)


def _components_shape_matches(state: RandomProjectionState) -> bool:
    return tuple(state.components.shape) == (state.n_components, state.n_features_in)


def _inverse_shape_matches(state: RandomProjectionState) -> bool:
    if state.inverse_components is None:
        return not state.compute_inverse_components
    return tuple(state.inverse_components.shape) == (state.n_features_in, state.n_components)


def _johnson_lindenstrauss_min_dim(n_samples: int, eps: float) -> int:
    if eps <= 0.0 or eps >= 1.0:
        raise ValueError(f"The JL bound is defined for eps in ]0, 1[, got {eps!r}")
    if n_samples <= 0:
        raise ValueError(f"The JL bound is defined for n_samples greater than zero, got {n_samples!r}")
    denominator = (eps**2 / 2) - (eps**3 / 3)
    return int(np.asarray(4 * np.log(n_samples) / denominator).astype(np.int64))


def _resolve_n_components(n_components: int | str, eps: float, n_samples: int, n_features: int) -> int:
    if n_components == "auto":
        resolved = _johnson_lindenstrauss_min_dim(n_samples=n_samples, eps=eps)
        if resolved <= 0:
            raise ValueError(
                "eps=%f and n_samples=%d lead to a target dimension of %d which is invalid"
                % (eps, n_samples, resolved)
            )
        if resolved > n_features:
            raise ValueError(
                "eps=%f and n_samples=%d lead to a target dimension of %d which is larger than the original "
                "space with n_features=%d" % (eps, n_samples, resolved, n_features)
            )
        return resolved

    if n_components > n_features:
        warnings.warn(
            "The number of components is higher than the number of features: n_features < n_components "
            f"({n_features} < {n_components}).The dimensionality of the problem will not be reduced.",
            DataDimensionalityWarning,
        )
    return int(n_components)


def _check_input_size(n_components: int, n_features: int) -> None:
    if n_components <= 0:
        raise ValueError("n_components must be strictly positive, got %d" % n_components)
    if n_features <= 0:
        raise ValueError("n_features must be strictly positive, got %d" % n_features)


def _check_density(density: float | str, n_features: int) -> float:
    if density == "auto":
        return float(1 / np.sqrt(n_features))
    if density <= 0 or density > 1:
        raise ValueError("Expected density in range ]0, 1], got: %r" % density)
    return float(density)


def _gaussian_random_matrix(n_components: int, n_features: int, random_state: object | None = None) -> NDArray[np.float64]:
    _check_input_size(n_components, n_features)
    rng = check_random_state(random_state)
    return rng.normal(loc=0.0, scale=1.0 / np.sqrt(n_components), size=(n_components, n_features))


def _sparse_random_matrix(
    n_components: int,
    n_features: int,
    density: float | str = "auto",
    random_state: object | None = None,
) -> MatrixLike:
    _check_input_size(n_components, n_features)
    density = _check_density(density, n_features)
    rng = check_random_state(random_state)

    if density == 1:
        components = rng.binomial(1, 0.5, (n_components, n_features)) * 2 - 1
        return 1 / np.sqrt(n_components) * components

    indices = []
    offset = 0
    indptr = [offset]
    for _ in range(n_components):
        n_nonzero_i = rng.binomial(n_features, density)
        indices_i = sample_without_replacement(n_features, n_nonzero_i, random_state=rng)
        indices.append(indices_i)
        offset += n_nonzero_i
        indptr.append(offset)

    data = rng.binomial(1, 0.5, size=sum(len(row) for row in indices)) * 2 - 1
    components = sp.csr_matrix((data, np.concatenate(indices), indptr), shape=(n_components, n_features))
    return np.sqrt(1 / density) / np.sqrt(n_components) * components


def _compute_inverse_components(components: MatrixLike) -> NDArray[np.float64]:
    dense_components = components.toarray() if sp.issparse(components) else components
    return linalg.pinv(dense_components, check_finite=False)


@register_atom(witness_gaussian_random_projection_fit)
@icontract.require(lambda X: _is_2d(X), "X must be a 2D matrix")
@icontract.require(lambda X: _row_count(X) >= 1, "X must contain at least one sample")
@icontract.require(lambda n_components: _valid_n_components(n_components), "n_components must be a positive integer or 'auto'")
@icontract.require(lambda n_components, eps: _valid_fit_eps(n_components, eps), "eps must be positive and in (0, 1) for auto components")
@icontract.ensure(lambda result: result.projection_kind == "gaussian", "state must record Gaussian projection kind")
@icontract.ensure(lambda result: _components_shape_matches(result), "components shape must match fitted dimensions")
@icontract.ensure(lambda result: _inverse_shape_matches(result), "inverse components must match fitted dimensions when cached")
def gaussian_random_projection_fit(
    X: MatrixLike,
    n_components: int | str = "auto",
    eps: float = 0.1,
    compute_inverse_components: bool = False,
    random_state: object | None = None,
) -> RandomProjectionState:
    """Fit Gaussian random projection state from a training matrix shape."""
    checked_x = check_array(X, accept_sparse=("csr", "csc"), dtype=[np.float64, np.float32])
    n_samples, n_features = checked_x.shape
    resolved_components = _resolve_n_components(n_components, eps, n_samples, n_features)
    components = _gaussian_random_matrix(resolved_components, n_features, random_state=random_state).astype(
        checked_x.dtype,
        copy=False,
    )
    inverse_components = _compute_inverse_components(components) if compute_inverse_components else None
    return RandomProjectionState(
        components=components,
        n_components=resolved_components,
        n_features_in=int(n_features),
        projection_kind="gaussian",
        compute_inverse_components=bool(compute_inverse_components),
        inverse_components=inverse_components,
    )


@register_atom(witness_gaussian_random_projection_transform)
@icontract.require(lambda X: _is_2d(X), "X must be a 2D matrix")
@icontract.require(lambda state: state.projection_kind == "gaussian", "state must be Gaussian projection state")
@icontract.require(lambda X, state: _feature_count(X) == state.n_features_in, "X feature count must match fitted state")
@icontract.ensure(lambda result, X, state: result.shape == (_row_count(X), state.n_components), "output shape must match projection")
def gaussian_random_projection_transform(
    X: MatrixLike,
    state: RandomProjectionState,
) -> NDArray[np.float64]:
    """Project data with a fitted Gaussian random projection state."""
    checked_x = check_array(X, accept_sparse=("csr", "csc"), dtype=[np.float64, np.float32])
    if checked_x.shape[1] != state.n_features_in:
        raise ValueError("X feature count does not match fitted state")
    return checked_x @ state.components.T


@register_atom(witness_sparse_random_projection_fit)
@icontract.require(lambda X: _is_2d(X), "X must be a 2D matrix")
@icontract.require(lambda X: _row_count(X) >= 1, "X must contain at least one sample")
@icontract.require(lambda n_components: _valid_n_components(n_components), "n_components must be a positive integer or 'auto'")
@icontract.require(lambda density: _valid_density(density), "density must be 'auto' or in (0, 1]")
@icontract.require(lambda n_components, eps: _valid_fit_eps(n_components, eps), "eps must be positive and in (0, 1) for auto components")
@icontract.ensure(lambda result: result.projection_kind == "sparse", "state must record sparse projection kind")
@icontract.ensure(lambda result: result.density is not None and 0.0 < result.density <= 1.0, "state density must be in (0, 1]")
@icontract.ensure(lambda result: _components_shape_matches(result), "components shape must match fitted dimensions")
@icontract.ensure(lambda result: _inverse_shape_matches(result), "inverse components must match fitted dimensions when cached")
def sparse_random_projection_fit(
    X: MatrixLike,
    n_components: int | str = "auto",
    density: float | str = "auto",
    eps: float = 0.1,
    dense_output: bool = False,
    compute_inverse_components: bool = False,
    random_state: object | None = None,
) -> RandomProjectionState:
    """Fit sparse random projection state from a training matrix shape."""
    checked_x = check_array(X, accept_sparse=("csr", "csc"), dtype=[np.float64, np.float32])
    n_samples, n_features = checked_x.shape
    resolved_components = _resolve_n_components(n_components, eps, n_samples, n_features)
    resolved_density = _check_density(density, n_features)
    components = _sparse_random_matrix(
        resolved_components,
        n_features,
        density=resolved_density,
        random_state=random_state,
    ).astype(checked_x.dtype, copy=False)
    inverse_components = _compute_inverse_components(components) if compute_inverse_components else None
    return RandomProjectionState(
        components=components,
        n_components=resolved_components,
        n_features_in=int(n_features),
        projection_kind="sparse",
        compute_inverse_components=bool(compute_inverse_components),
        inverse_components=inverse_components,
        density=resolved_density,
        dense_output=bool(dense_output),
    )


@register_atom(witness_sparse_random_projection_transform)
@icontract.require(lambda X: _is_2d(X), "X must be a 2D matrix")
@icontract.require(lambda state: state.projection_kind == "sparse", "state must be sparse projection state")
@icontract.require(lambda X, state: _feature_count(X) == state.n_features_in, "X feature count must match fitted state")
@icontract.ensure(lambda result, X, state: result.shape == (_row_count(X), state.n_components), "output shape must match projection")
def sparse_random_projection_transform(
    X: MatrixLike,
    state: RandomProjectionState,
) -> MatrixLike:
    """Project data with a fitted sparse random projection state."""
    checked_x = check_array(X, accept_sparse=("csr", "csc"), dtype=[np.float64, np.float32])
    if checked_x.shape[1] != state.n_features_in:
        raise ValueError("X feature count does not match fitted state")
    return safe_sparse_dot(checked_x, state.components.T, dense_output=state.dense_output)


@register_atom(witness_random_projection_inverse_transform)
@icontract.require(lambda X: _is_2d(X), "X must be a 2D matrix")
@icontract.require(lambda X, state: _feature_count(X) == state.n_components, "X feature count must match projection components")
@icontract.require(lambda state: _components_shape_matches(state), "components shape must match fitted state")
@icontract.ensure(lambda result, X, state: result.shape == (_row_count(X), state.n_features_in), "inverse output shape must match original feature count")
def random_projection_inverse_transform(
    X: MatrixLike,
    state: RandomProjectionState,
) -> NDArray[np.float64]:
    """Project coordinates from random projection space back to input space."""
    checked_x = check_array(X, accept_sparse=("csr", "csc"), dtype=[np.float64, np.float32])
    if checked_x.shape[1] != state.n_components:
        raise ValueError("X feature count does not match fitted projection components")
    inverse_components = (
        state.inverse_components if state.compute_inverse_components and state.inverse_components is not None else _compute_inverse_components(state.components)
    )
    return checked_x @ inverse_components.T
