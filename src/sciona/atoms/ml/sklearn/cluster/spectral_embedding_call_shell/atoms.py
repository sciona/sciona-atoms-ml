"""Spectral clustering embedding-call atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_spectral_fit_embedding_call_kwargs,
    witness_spectral_fit_embedding_drop_first,
    witness_spectral_fit_embedding_random_state,
)

def _positive_int(value: object) -> bool:
    return bool(isinstance(value, int) and not isinstance(value, bool) and value >= 1)

def _random_state_valid(value: object) -> bool:
    return bool(value is None or (isinstance(value, int) and not isinstance(value, bool) and value >= 0))

def _eigen_solver_valid(value: object) -> bool:
    return bool(value is None or isinstance(value, str))

def _eigen_tol_valid(value: object) -> bool:
    return bool(
        value == "auto"
        or (
            isinstance(value, (int, float, np.integer, np.floating))
            and not isinstance(value, bool)
            and np.isfinite(float(value))
            and float(value) >= 0.0
        )
    )

def _random_state_result_valid(result: object) -> bool:
    return isinstance(result, np.random.RandomState)

def _same_rng_sequence(result: object, random_state: int | None) -> bool:
    from sklearn.utils import check_random_state
    if not isinstance(result, np.random.RandomState):
        return False
    expected = check_random_state(random_state)
    lhs = result.get_state()
    rhs = expected.get_state()
    return bool(
        lhs[0] == rhs[0]
        and np.array_equal(lhs[1], rhs[1])
        and lhs[2] == rhs[2]
        and lhs[3] == rhs[3]
        and lhs[4] == rhs[4]
    )

def _embedding_kwargs_valid(
    result: object,
    n_components: int,
    eigen_solver: str | None,
    random_state: int | None,
    eigen_tol: float | str,
) -> bool:
    if not isinstance(result, dict):
        return False
    if set(result) != {"n_components", "eigen_solver", "random_state", "eigen_tol", "drop_first"}:
        return False
    if result["n_components"] != int(n_components):
        return False
    if result["eigen_solver"] != eigen_solver:
        return False
    if result["eigen_tol"] != eigen_tol:
        return False
    if result["drop_first"] is not False:
        return False
    return _same_rng_sequence(result["random_state"], random_state)

@register_atom(witness_spectral_fit_embedding_random_state)
@icontract.require(lambda random_state=None: _random_state_valid(random_state), "random_state must be None or a nonnegative integer")
@icontract.ensure(lambda result: _random_state_result_valid(result), "result must be a numpy RandomState")
@icontract.ensure(lambda result, random_state=None: _same_rng_sequence(result, random_state), "result must match sklearn's normalized random state")
def spectral_fit_embedding_random_state(random_state: int | None = None) -> np.random.RandomState:
    from sklearn.utils import check_random_state
    """Normalize SpectralClustering.fit's random_state for the _spectral_embedding call."""
    return check_random_state(random_state)

@register_atom(witness_spectral_fit_embedding_drop_first)
@icontract.require(lambda parent_drop_first=True: isinstance(parent_drop_first, bool), "parent_drop_first must be boolean")
@icontract.ensure(lambda result: isinstance(result, bool) and result is False, "drop_first must be fixed to False")
def spectral_fit_embedding_drop_first(parent_drop_first: bool = True) -> bool:
    """Expose the fixed drop_first=False argument used by SpectralClustering.fit."""
    del parent_drop_first
    return False

@register_atom(witness_spectral_fit_embedding_call_kwargs)
@icontract.require(lambda n_components: _positive_int(n_components), "n_components must be a positive integer")
@icontract.require(lambda eigen_solver=None: _eigen_solver_valid(eigen_solver), "eigen_solver must be None or a string")
@icontract.require(lambda random_state=None: _random_state_valid(random_state), "random_state must be None or a nonnegative integer")
@icontract.require(lambda eigen_tol="auto": _eigen_tol_valid(eigen_tol), "eigen_tol must be 'auto' or a nonnegative finite scalar")
@icontract.ensure(
    lambda result, n_components, eigen_solver=None, random_state=None, eigen_tol="auto": _embedding_kwargs_valid(
        result,
        n_components,
        eigen_solver,
        random_state,
        eigen_tol,
    ),
    "result must match SpectralClustering's _spectral_embedding kwargs",
)
def spectral_fit_embedding_call_kwargs(
    n_components: int,
    eigen_solver: str | None,
    random_state: int | None = None,
    eigen_tol: float | str = "auto",
) -> dict[str, object]:
    """Resolve the keyword arguments SpectralClustering.fit passes into _spectral_embedding."""
    return {
        "n_components": int(n_components),
        "eigen_solver": eigen_solver,
        "random_state": spectral_fit_embedding_random_state(random_state),
        "eigen_tol": eigen_tol,
        "drop_first": spectral_fit_embedding_drop_first(),
    }
