"""Scheduling and bookkeeping atoms adapted from scikit-learn t-SNE."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray
from scipy import linalg
from scipy.sparse import csr_matrix, issparse

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_tsne_early_exaggeration_scale,
    witness_tsne_early_exaggeration_unscale,
    witness_tsne_gradient_descent_buffers,
    witness_tsne_gradient_descent_compute_error,
    witness_tsne_gradient_descent_convergence,
    witness_tsne_stage_two_required,
)

TsneBuffers = tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]
TsneConvergence = tuple[float, int, float, bool]
ProbabilityLike = NDArray[np.float64] | csr_matrix


def _positive_int(value: int) -> bool:
    return bool(isinstance(value, int) and not isinstance(value, bool) and value >= 1)


def _nonnegative_int(value: int) -> bool:
    return bool(isinstance(value, int) and not isinstance(value, bool) and value >= 0)


def _finite_nonnegative_scalar(value: float) -> bool:
    return bool(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and np.isfinite(float(value))
        and float(value) >= 0.0
    )


def _finite_vector(values: NDArray[np.float64]) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 1 and array.size >= 1 and np.all(np.isfinite(array)))


def _probability_like(values: ProbabilityLike) -> bool:
    if issparse(values):
        return bool(
            isinstance(values, csr_matrix)
            and values.shape[0] >= 1
            and values.shape[1] >= 1
            and np.all(np.isfinite(values.data))
            and np.all(values.data >= 0.0)
        )
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim in {1, 2} and array.size >= 1 and np.all(np.isfinite(array)) and np.all(array >= 0.0))


def _same_probability_like(result: ProbabilityLike, values: ProbabilityLike) -> bool:
    if issparse(values):
        return bool(
            isinstance(result, csr_matrix)
            and result.shape == values.shape
            and np.all(np.isfinite(result.data))
            and np.all(result.data >= 0.0)
        )
    array = np.asarray(result, dtype=np.float64)
    source = np.asarray(values, dtype=np.float64)
    return bool(array.shape == source.shape and np.all(np.isfinite(array)) and np.all(array >= 0.0))


def _buffers_valid(result: TsneBuffers, p0: NDArray[np.float64]) -> bool:
    if not (isinstance(result, tuple) and len(result) == 3):
        return False
    params, update, gains = result
    source = np.asarray(p0, dtype=np.float64).ravel()
    return bool(
        np.asarray(params).shape == source.shape
        and np.asarray(update).shape == source.shape
        and np.asarray(gains).shape == source.shape
        and np.all(np.isfinite(params))
        and np.all(np.isfinite(update))
        and np.all(np.isfinite(gains))
        and np.allclose(params, source)
        and np.allclose(update, 0.0)
        and np.allclose(gains, 1.0)
    )


def _convergence_result_valid(result: TsneConvergence, best_iter: int, iteration: int) -> bool:
    if not (isinstance(result, tuple) and len(result) == 4):
        return False
    new_best_error, new_best_iter, grad_norm, should_stop = result
    return bool(
        _finite_nonnegative_scalar(float(new_best_error))
        and isinstance(new_best_iter, int)
        and best_iter <= new_best_iter <= iteration
        and _finite_nonnegative_scalar(float(grad_norm))
        and isinstance(should_stop, bool)
    )


@register_atom(witness_tsne_gradient_descent_buffers)
@icontract.require(lambda p0: _finite_vector(p0), "p0 must be a finite nonempty vector")
@icontract.ensure(lambda result, p0: _buffers_valid(result, p0), "buffer initialization must preserve shape and produce zero updates with unit gains")
def tsne_gradient_descent_buffers(p0: NDArray[np.float64]) -> TsneBuffers:
    """Flatten parameters and allocate the update and gain buffers sklearn uses."""
    params = np.asarray(p0, dtype=np.float64).copy().ravel()
    return params, np.zeros_like(params), np.ones_like(params)


@register_atom(witness_tsne_gradient_descent_compute_error)
@icontract.require(lambda iteration: _nonnegative_int(iteration), "iteration must be a nonnegative integer")
@icontract.require(lambda max_iter: _positive_int(max_iter), "max_iter must be a positive integer")
@icontract.require(lambda iteration, max_iter: iteration < max_iter, "iteration must be smaller than max_iter")
@icontract.require(lambda n_iter_check: _positive_int(n_iter_check), "n_iter_check must be a positive integer")
@icontract.ensure(lambda result: isinstance(result, bool), "compute-error flag must be boolean")
def tsne_gradient_descent_compute_error(
    iteration: int,
    max_iter: int,
    *,
    n_iter_check: int = 1,
) -> bool:
    """Return whether sklearn asks the objective for the scalar error at this iteration."""
    check_convergence = (iteration + 1) % n_iter_check == 0
    return bool(check_convergence or iteration == max_iter - 1)


@register_atom(witness_tsne_gradient_descent_convergence)
@icontract.require(lambda error: _finite_nonnegative_scalar(error), "error must be finite and nonnegative")
@icontract.require(lambda grad: _finite_vector(grad), "grad must be a finite nonempty vector")
@icontract.require(lambda iteration: _nonnegative_int(iteration), "iteration must be a nonnegative integer")
@icontract.require(lambda best_error: _finite_nonnegative_scalar(best_error), "best_error must be finite and nonnegative")
@icontract.require(lambda best_iter: _nonnegative_int(best_iter), "best_iter must be a nonnegative integer")
@icontract.require(lambda iteration, best_iter: best_iter <= iteration, "best_iter must not exceed iteration")
@icontract.require(lambda n_iter_without_progress: isinstance(n_iter_without_progress, int) and not isinstance(n_iter_without_progress, bool) and n_iter_without_progress >= -1, "n_iter_without_progress must be an integer >= -1")
@icontract.require(lambda min_grad_norm: _finite_nonnegative_scalar(min_grad_norm), "min_grad_norm must be finite and nonnegative")
@icontract.ensure(lambda result, best_iter, iteration: _convergence_result_valid(result, best_iter, iteration), "convergence bookkeeping must stay finite and bounded")
def tsne_gradient_descent_convergence(
    error: float,
    grad: NDArray[np.float64],
    iteration: int,
    best_error: float,
    best_iter: int,
    *,
    n_iter_without_progress: int = 300,
    min_grad_norm: float = 1e-7,
) -> TsneConvergence:
    """Update best-error bookkeeping and decide whether sklearn would stop."""
    grad_norm = float(linalg.norm(np.asarray(grad, dtype=np.float64)))
    next_best_error = float(best_error)
    next_best_iter = int(best_iter)
    should_stop = False

    if error < best_error:
        next_best_error = float(error)
        next_best_iter = int(iteration)
    elif iteration - best_iter > n_iter_without_progress:
        should_stop = True

    if grad_norm <= min_grad_norm:
        should_stop = True

    return next_best_error, next_best_iter, grad_norm, should_stop


@register_atom(witness_tsne_early_exaggeration_scale)
@icontract.require(lambda probabilities: _probability_like(probabilities), "probabilities must be finite nonnegative dense or CSR data")
@icontract.require(lambda early_exaggeration: _finite_nonnegative_scalar(early_exaggeration) and float(early_exaggeration) >= 1.0, "early_exaggeration must be finite and at least 1")
@icontract.ensure(lambda result, probabilities: _same_probability_like(result, probabilities), "scaled probabilities must preserve storage layout and shape")
def tsne_early_exaggeration_scale(
    probabilities: ProbabilityLike,
    early_exaggeration: float,
) -> ProbabilityLike:
    """Apply sklearn's early-exaggeration scaling to dense or CSR probabilities."""
    if issparse(probabilities):
        scaled = probabilities.copy().tocsr()
        scaled.data *= float(early_exaggeration)
        return scaled
    return np.asarray(probabilities, dtype=np.float64).copy() * float(early_exaggeration)


@register_atom(witness_tsne_early_exaggeration_unscale)
@icontract.require(lambda probabilities: _probability_like(probabilities), "probabilities must be finite nonnegative dense or CSR data")
@icontract.require(lambda early_exaggeration: _finite_nonnegative_scalar(early_exaggeration) and float(early_exaggeration) >= 1.0, "early_exaggeration must be finite and at least 1")
@icontract.ensure(lambda result, probabilities: _same_probability_like(result, probabilities), "unscaled probabilities must preserve storage layout and shape")
def tsne_early_exaggeration_unscale(
    probabilities: ProbabilityLike,
    early_exaggeration: float,
) -> ProbabilityLike:
    """Remove sklearn's early-exaggeration scaling from dense or CSR probabilities."""
    if issparse(probabilities):
        unscaled = probabilities.copy().tocsr()
        unscaled.data /= float(early_exaggeration)
        return unscaled
    return np.asarray(probabilities, dtype=np.float64).copy() / float(early_exaggeration)


@register_atom(witness_tsne_stage_two_required)
@icontract.require(lambda iteration: _nonnegative_int(iteration), "iteration must be a nonnegative integer")
@icontract.require(lambda exploration_max_iter: _positive_int(exploration_max_iter), "exploration_max_iter must be a positive integer")
@icontract.require(lambda max_iter: _positive_int(max_iter), "max_iter must be a positive integer")
@icontract.require(lambda max_iter, exploration_max_iter: max_iter >= exploration_max_iter, "max_iter must be at least exploration_max_iter")
@icontract.ensure(lambda result: isinstance(result, bool), "stage-two predicate must be boolean")
def tsne_stage_two_required(
    iteration: int,
    exploration_max_iter: int,
    max_iter: int,
) -> bool:
    """Return whether sklearn enters the second t-SNE optimization stage."""
    remaining = max_iter - exploration_max_iter
    return bool(iteration < exploration_max_iter or remaining > 0)
