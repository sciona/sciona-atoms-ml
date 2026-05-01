"""GraphicalLassoCV fit-setup helper atoms adapted from scikit-learn."""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Real

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_graphical_lasso_cv_explicit_alphas,
    witness_graphical_lasso_cv_inner_verbose,
    witness_graphical_lasso_cv_location,
    witness_graphical_lasso_cv_refinement_count,
    witness_graphical_lasso_cv_use_explicit_alphas,
)


def _finite_matrix(values: object) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 2 and array.shape[0] >= 1 and array.shape[1] >= 2 and np.all(np.isfinite(array)))


def _bool(value: object) -> bool:
    return isinstance(value, bool)


def _location_valid(result: object, X: object) -> bool:
    values = np.asarray(result, dtype=np.float64)
    matrix = np.asarray(X, dtype=np.float64)
    return bool(values.shape == (matrix.shape[1],) and np.all(np.isfinite(values)))


def _integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _positive_int(value: object) -> bool:
    return _integer(value) and int(value) >= 1


def _arraylike_not_scalar(values: object) -> bool:
    if isinstance(values, (str, bytes)):
        return False
    if np.isscalar(values):
        return False
    return isinstance(values, Sequence) or hasattr(values, "__array__")


def _nonnegative_real_scalar(value: object) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and np.isfinite(float(value)) and float(value) >= 0.0


def _explicit_alpha_vector_valid(result: object) -> bool:
    try:
        values = np.asarray(result, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(values.ndim == 1 and values.shape[0] >= 1 and np.all(np.isfinite(values)) and np.all(values >= 0.0))


@register_atom(witness_graphical_lasso_cv_location)
@icontract.require(lambda X: _finite_matrix(X), "X must be a finite 2D sample matrix with at least two features")
@icontract.require(lambda assume_centered: _bool(assume_centered), "assume_centered must be boolean")
@icontract.ensure(lambda result, X: _location_valid(result, X), "location must be a finite feature-length vector")
def graphical_lasso_cv_location(
    X: NDArray[np.float64],
    assume_centered: bool,
) -> NDArray[np.float64]:
    """Resolve GraphicalLassoCV's fitted location vector before path solving."""
    values = np.asarray(X, dtype=np.float64)
    if assume_centered:
        return np.zeros(values.shape[1], dtype=np.float64)
    return np.asarray(values.mean(axis=0), dtype=np.float64)


@register_atom(witness_graphical_lasso_cv_inner_verbose)
@icontract.require(lambda verbose: _integer(verbose), "verbose must be an integer")
@icontract.ensure(lambda result: _integer(result), "inner verbosity must be an integer")
def graphical_lasso_cv_inner_verbose(
    verbose: int,
) -> int:
    """Resolve GraphicalLassoCV's inner path verbosity level."""
    return max(0, int(verbose) - 1)


@register_atom(witness_graphical_lasso_cv_use_explicit_alphas)
@icontract.require(lambda alphas: True, "alphas may be any object passed into GraphicalLassoCV")
@icontract.ensure(lambda result: _bool(result), "explicit-alpha mode predicate must be boolean")
def graphical_lasso_cv_use_explicit_alphas(
    alphas: object,
) -> bool:
    """Decide whether GraphicalLassoCV treats alphas as an explicit sequence."""
    return _arraylike_not_scalar(alphas)


@register_atom(witness_graphical_lasso_cv_explicit_alphas)
@icontract.require(lambda alphas: _arraylike_not_scalar(alphas), "alphas must be a non-scalar array-like sequence")
@icontract.require(lambda alphas: all(_nonnegative_real_scalar(alpha) for alpha in alphas), "each explicit alpha must be a finite nonnegative scalar")
@icontract.ensure(lambda result: _explicit_alpha_vector_valid(result), "explicit alphas must form a finite nonnegative 1D vector")
def graphical_lasso_cv_explicit_alphas(
    alphas: object,
) -> NDArray[np.float64]:
    """Validate and materialize GraphicalLassoCV's explicit alpha sequence."""
    return np.asarray(list(alphas), dtype=np.float64)


@register_atom(witness_graphical_lasso_cv_refinement_count)
@icontract.require(lambda use_explicit_alphas: _bool(use_explicit_alphas), "use_explicit_alphas must be boolean")
@icontract.require(lambda n_refinements: _positive_int(n_refinements), "n_refinements must be a positive integer")
@icontract.ensure(lambda result: _positive_int(result), "refinement count must be a positive integer")
def graphical_lasso_cv_refinement_count(
    use_explicit_alphas: bool,
    n_refinements: int,
) -> int:
    """Resolve GraphicalLassoCV's refinement count from alpha input mode."""
    if use_explicit_alphas:
        return 1
    return int(n_refinements)
