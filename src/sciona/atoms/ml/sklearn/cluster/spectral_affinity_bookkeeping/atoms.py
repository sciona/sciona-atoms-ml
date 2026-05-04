"""Spectral clustering affinity-bookkeeping atoms adapted from scikit-learn."""

from __future__ import annotations

from collections.abc import Mapping

import icontract
import numpy as np
import scipy.sparse as sp

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_spectral_fit_pairwise_kernel_params,
    witness_spectral_fit_symmetric_connectivity,
    witness_spectral_fit_use_nearest_neighbors,
    witness_spectral_fit_use_pairwise_kernel_hyperparameters,
    witness_spectral_fit_use_precomputed_affinity,
    witness_spectral_fit_use_precomputed_nearest_neighbors,
)


def _nonempty_string(value: object) -> bool:
    return bool(isinstance(value, str) and value != "")


def _bool(value: object) -> bool:
    return isinstance(value, bool)


def _finite_real_scalar(value: object) -> bool:
    return bool(
        isinstance(value, (int, float, np.integer, np.floating))
        and not isinstance(value, bool)
        and np.isfinite(float(value))
    )


def _kernel_params_mapping(value: object) -> bool:
    if value is None:
        return True
    if not isinstance(value, Mapping):
        return False
    return all(isinstance(key, str) and key != "" for key in value)


def _callable_or_nonempty_string(value: object) -> bool:
    return callable(value) or _nonempty_string(value)


def _to_dense(values: object) -> np.ndarray:
    if sp.issparse(values):
        return values.toarray()
    return np.asarray(values)


def _square_finite_connectivity(value: object) -> bool:
    if sp.issparse(value):
        return bool(
            value.ndim == 2
            and value.shape[0] >= 1
            and value.shape[0] == value.shape[1]
            and np.all(np.isfinite(value.data))
        )
    array = np.asarray(value)
    return bool(
        array.ndim == 2
        and array.shape[0] >= 1
        and array.shape[0] == array.shape[1]
        and np.all(np.isfinite(array))
    )


def _same_mapping_items(result: object, kernel_params: Mapping[str, object] | None) -> bool:
    if not isinstance(result, dict):
        return False
    expected = {} if kernel_params is None else dict(kernel_params)
    return result == expected


def _resolved_kernel_params_valid(
    result: object,
    kernel_params: Mapping[str, object] | None,
    use_kernel_hyperparameters: bool,
    gamma: float,
    degree: float,
    coef0: float,
) -> bool:
    if not isinstance(result, dict):
        return False
    expected = {} if kernel_params is None else dict(kernel_params)
    if use_kernel_hyperparameters:
        expected["gamma"] = gamma
        expected["degree"] = degree
        expected["coef0"] = coef0
    return result == expected


def _symmetric_connectivity_valid(result: object, connectivity: object) -> bool:
    if sp.issparse(connectivity):
        if not sp.issparse(result):
            return False
    elif sp.issparse(result):
        return False
    lhs = _to_dense(result)
    rhs = 0.5 * (_to_dense(connectivity) + _to_dense(connectivity).T)
    return bool(lhs.shape == rhs.shape and np.allclose(lhs, rhs) and np.allclose(lhs, lhs.T))


@register_atom(witness_spectral_fit_use_nearest_neighbors)
@icontract.require(lambda affinity: _nonempty_string(affinity), "affinity must be a nonempty string")
@icontract.ensure(lambda result: _bool(result), "result must be boolean")
def spectral_fit_use_nearest_neighbors(affinity: str) -> bool:
    """Return whether SpectralClustering.fit takes the nearest-neighbors affinity branch."""
    return affinity == "nearest_neighbors"


@register_atom(witness_spectral_fit_use_precomputed_nearest_neighbors)
@icontract.require(lambda affinity: _nonempty_string(affinity), "affinity must be a nonempty string")
@icontract.ensure(lambda result: _bool(result), "result must be boolean")
def spectral_fit_use_precomputed_nearest_neighbors(affinity: str) -> bool:
    """Return whether SpectralClustering.fit takes the precomputed-nearest-neighbors branch."""
    return affinity == "precomputed_nearest_neighbors"


@register_atom(witness_spectral_fit_use_precomputed_affinity)
@icontract.require(lambda affinity: _nonempty_string(affinity), "affinity must be a nonempty string")
@icontract.ensure(lambda result: _bool(result), "result must be boolean")
def spectral_fit_use_precomputed_affinity(affinity: str) -> bool:
    """Return whether SpectralClustering.fit uses the supplied precomputed affinity matrix directly."""
    return affinity == "precomputed"


@register_atom(witness_spectral_fit_use_pairwise_kernel_hyperparameters)
@icontract.require(
    lambda affinity: _callable_or_nonempty_string(affinity),
    "affinity must be a callable or a nonempty string",
)
@icontract.ensure(lambda result: _bool(result), "result must be boolean")
def spectral_fit_use_pairwise_kernel_hyperparameters(affinity: object) -> bool:
    """Return whether SpectralClustering injects gamma, degree, and coef0 into pairwise_kernels."""
    return not callable(affinity)


@register_atom(witness_spectral_fit_pairwise_kernel_params)
@icontract.require(
    lambda kernel_params=None: _kernel_params_mapping(kernel_params),
    "kernel_params must be None or a mapping with nonempty string keys",
)
@icontract.require(
    lambda use_kernel_hyperparameters: _bool(use_kernel_hyperparameters),
    "use_kernel_hyperparameters must be boolean",
)
@icontract.require(lambda gamma: _finite_real_scalar(gamma), "gamma must be a finite real scalar")
@icontract.require(lambda degree: _finite_real_scalar(degree), "degree must be a finite real scalar")
@icontract.require(lambda coef0: _finite_real_scalar(coef0), "coef0 must be a finite real scalar")
@icontract.ensure(
    lambda result, kernel_params, use_kernel_hyperparameters, gamma, degree, coef0: _resolved_kernel_params_valid(
        result,
        kernel_params,
        use_kernel_hyperparameters,
        gamma,
        degree,
        coef0,
    ),
    "result must match SpectralClustering's resolved pairwise-kernel parameters",
)
def spectral_fit_pairwise_kernel_params(
    kernel_params: Mapping[str, object] | None,
    use_kernel_hyperparameters: bool,
    gamma: float,
    degree: float,
    coef0: float,
) -> dict[str, object]:
    """Resolve the pairwise_kernels parameter mapping used by SpectralClustering.fit."""
    params = {} if kernel_params is None else dict(kernel_params)
    if use_kernel_hyperparameters:
        params["gamma"] = gamma
        params["degree"] = degree
        params["coef0"] = coef0
    return params


@register_atom(witness_spectral_fit_symmetric_connectivity)
@icontract.require(
    lambda connectivity: _square_finite_connectivity(connectivity),
    "connectivity must be a nonempty square finite dense or sparse matrix",
)
@icontract.ensure(
    lambda result, connectivity: _symmetric_connectivity_valid(result, connectivity),
    "result must equal 0.5 * (connectivity + connectivity.T) with matching sparse-or-dense form",
)
def spectral_fit_symmetric_connectivity(connectivity: object) -> object:
    """Symmetrize the nearest-neighbor connectivity matrix used for SpectralClustering.affinity_matrix_."""
    return 0.5 * (connectivity + connectivity.T)
