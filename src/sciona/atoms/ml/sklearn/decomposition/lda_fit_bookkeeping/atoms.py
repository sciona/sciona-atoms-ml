"""LatentDirichletAllocation fit bookkeeping helpers adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_lda_batch_bounds,
    witness_lda_check_nonnegative_dtype_names,
    witness_lda_fit_converged,
    witness_lda_fit_evaluate_iteration_due,
    witness_lda_fit_use_online_batches,
    witness_lda_partial_fit_first_call,
    witness_lda_partial_fit_require_matching_feature_count,
)


def _positive_int(value: object) -> bool:
    return bool(isinstance(value, int) and not isinstance(value, bool) and value >= 1)


def _nonnegative_int(value: object) -> bool:
    return bool(isinstance(value, int) and not isinstance(value, bool) and value >= 0)


def _finite_positive_scalar(value: object) -> bool:
    return bool(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and np.isfinite(float(value))
        and float(value) > 0.0
    )


def _finite_nonnegative_scalar(value: object) -> bool:
    return bool(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and np.isfinite(float(value))
        and float(value) >= 0.0
    )


def _optional_positive_scalar(value: object) -> bool:
    return value is None or _finite_positive_scalar(value)


def _learning_method_valid(value: object) -> bool:
    return value in {"online", "batch"}


def _dtype_name_valid(value: object) -> bool:
    return value in {"float32", "float64"}


def _bool_result(value: object) -> bool:
    return isinstance(value, bool)


def _batch_bounds_valid(result: object, n_samples: int, batch_size: int) -> bool:
    values = np.asarray(result)
    if not (
        values.ndim == 2
        and values.shape[1] == 2
        and np.issubdtype(values.dtype, np.integer)
        and values.shape[0] == (n_samples + batch_size - 1) // batch_size
    ):
        return False
    starts = values[:, 0]
    stops = values[:, 1]
    if starts[0] != 0 or stops[-1] != n_samples:
        return False
    if np.any(starts < 0) or np.any(stops <= starts) or np.any(stops > n_samples):
        return False
    if np.any(starts[1:] != stops[:-1]):
        return False
    return True


def _dtype_names_valid(result: object, reset_n_features: bool) -> bool:
    values = np.asarray(result, dtype=object)
    allowed = {"float32", "float64"}
    if values.ndim != 1 or values.shape[0] < 1:
        return False
    if any(value not in allowed for value in values.tolist()):
        return False
    if reset_n_features:
        return bool(values.shape[0] == 2 and set(values.tolist()) == allowed)
    return bool(values.shape[0] == 1)


@register_atom(witness_lda_fit_use_online_batches)
@icontract.require(lambda learning_method: _learning_method_valid(learning_method), "learning_method must be 'online' or 'batch'")
@icontract.ensure(lambda result: _bool_result(result), "online-batch flag must be boolean")
def lda_fit_use_online_batches(learning_method: str) -> bool:
    """Return whether LDA fit should iterate over minibatches for the given method."""
    return learning_method == "online"


@register_atom(witness_lda_fit_evaluate_iteration_due)
@icontract.require(lambda iteration_index: _nonnegative_int(iteration_index), "iteration_index must be a nonnegative integer")
@icontract.require(lambda evaluate_every: _positive_int(evaluate_every), "evaluate_every must be a positive integer")
@icontract.ensure(lambda result: _bool_result(result), "evaluation-cadence flag must be boolean")
def lda_fit_evaluate_iteration_due(
    iteration_index: int,
    *,
    evaluate_every: int,
) -> bool:
    """Return whether this zero-based LDA fit iteration should evaluate perplexity."""
    return (int(iteration_index) + 1) % int(evaluate_every) == 0


@register_atom(witness_lda_fit_converged)
@icontract.require(lambda last_bound: _optional_positive_scalar(last_bound), "last_bound must be None or a finite positive scalar")
@icontract.require(lambda bound: _finite_positive_scalar(bound), "bound must be a finite positive scalar")
@icontract.require(lambda perp_tol: _finite_nonnegative_scalar(perp_tol), "perp_tol must be a finite nonnegative scalar")
@icontract.ensure(lambda result: _bool_result(result), "convergence flag must be boolean")
def lda_fit_converged(
    last_bound: float | None,
    bound: float,
    *,
    perp_tol: float,
) -> bool:
    """Return sklearn's LDA perplexity-improvement stopping predicate."""
    return bool(last_bound and abs(float(last_bound) - float(bound)) < float(perp_tol))


@register_atom(witness_lda_batch_bounds)
@icontract.require(lambda n_samples: _positive_int(n_samples), "n_samples must be a positive integer")
@icontract.require(lambda batch_size: _positive_int(batch_size), "batch_size must be a positive integer")
@icontract.ensure(lambda result, n_samples, batch_size: _batch_bounds_valid(result, n_samples, batch_size), "batch bounds must partition samples into sklearn-style contiguous minibatches")
def lda_batch_bounds(
    n_samples: int,
    *,
    batch_size: int,
) -> NDArray[np.int64]:
    """Return the contiguous minibatch start-stop bounds sklearn uses for LDA."""
    bounds = [
        (start, min(start + int(batch_size), int(n_samples)))
        for start in range(0, int(n_samples), int(batch_size))
    ]
    return np.asarray(bounds, dtype=np.int64)


@register_atom(witness_lda_partial_fit_first_call)
@icontract.require(lambda has_components: isinstance(has_components, bool), "has_components must be boolean")
@icontract.ensure(lambda result: _bool_result(result), "first-call flag must be boolean")
def lda_partial_fit_first_call(
    *,
    has_components: bool,
) -> bool:
    """Return whether LDA partial_fit is entering before components are initialized."""
    return not has_components


@register_atom(witness_lda_partial_fit_require_matching_feature_count)
@icontract.require(lambda n_features: _positive_int(n_features), "n_features must be a positive integer")
@icontract.require(lambda trained_feature_count: _positive_int(trained_feature_count), "trained_feature_count must be a positive integer")
@icontract.ensure(lambda result, n_features: result == n_features, "validated feature count must preserve the provided feature count")
def lda_partial_fit_require_matching_feature_count(
    n_features: int,
    *,
    trained_feature_count: int,
) -> int:
    """Require the incoming LDA partial_fit feature count to match the trained width."""
    if int(n_features) != int(trained_feature_count):
        raise ValueError(
            "The provided data has %d dimensions while the model was trained with feature size %d."
            % (int(n_features), int(trained_feature_count))
        )
    return int(n_features)


@register_atom(witness_lda_check_nonnegative_dtype_names)
@icontract.require(lambda reset_n_features: isinstance(reset_n_features, bool), "reset_n_features must be boolean")
@icontract.require(lambda components_dtype_name: _dtype_name_valid(components_dtype_name), "components_dtype_name must be 'float32' or 'float64'")
@icontract.ensure(lambda result, reset_n_features: _dtype_names_valid(result, reset_n_features), "dtype-name vector must match sklearn's reset-vs-fitted validation branch")
def lda_check_nonnegative_dtype_names(
    *,
    reset_n_features: bool,
    components_dtype_name: str = "float64",
) -> NDArray[np.object_]:
    """Return the dtype-name choices LDA uses before validating nonnegative input data."""
    if reset_n_features:
        return np.asarray(["float64", "float32"], dtype=object)
    return np.asarray([components_dtype_name], dtype=object)
