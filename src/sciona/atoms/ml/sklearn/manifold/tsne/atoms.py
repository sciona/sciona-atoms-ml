"""Deterministic exact-method t-SNE helper atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import pdist, squareform

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_tsne_exact_joint_probabilities,
    witness_tsne_exact_kl_divergence,
    witness_tsne_gradient_descent_update,
)

MACHINE_EPSILON = np.finfo(np.double).eps
TsneUpdate = tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]
TsneObjective = tuple[float, NDArray[np.float64]]


def _finite_vector(values: NDArray[np.float64]) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 1 and array.size >= 1 and np.all(np.isfinite(array)))


def _finite_square_matrix(values: NDArray[np.float64]) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 2 and array.shape[0] == array.shape[1] and array.shape[0] >= 2 and np.all(np.isfinite(array)))


def _nonnegative_square_matrix(values: NDArray[np.float64]) -> bool:
    return bool(_finite_square_matrix(values) and np.all(np.asarray(values, dtype=np.float64) >= 0.0))


def _condensed_length(n_samples: int) -> int:
    return n_samples * (n_samples - 1) // 2


def _joint_result_valid(result: NDArray[np.float64], conditional_probabilities: NDArray[np.float64]) -> bool:
    values = np.asarray(result, dtype=np.float64)
    n_samples = np.asarray(conditional_probabilities).shape[0]
    return bool(values.ndim == 1 and values.shape == (_condensed_length(n_samples),) and np.all(np.isfinite(values)) and np.all(values >= MACHINE_EPSILON))


def _positive_int(value: int) -> bool:
    return bool(isinstance(value, int) and not isinstance(value, bool) and value >= 1)


def _nonnegative_int(value: int) -> bool:
    return bool(isinstance(value, int) and not isinstance(value, bool) and value >= 0)


def _positive_finite(value: float) -> bool:
    return bool(isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(float(value)) and float(value) > 0.0)


def _unit_interval(value: float) -> bool:
    return bool(isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(float(value)) and 0.0 <= float(value) < 1.0)


def _kl_inputs_valid(
    params: NDArray[np.float64],
    P: NDArray[np.float64],
    degrees_of_freedom: int,
    n_samples: int,
    n_components: int,
    skip_num_points: int,
) -> bool:
    if not (_finite_vector(params) and _finite_vector(P)):
        return False
    if not (_positive_int(degrees_of_freedom) and _positive_int(n_samples) and _positive_int(n_components)):
        return False
    if not (_nonnegative_int(skip_num_points) and skip_num_points == 0):
        return False
    params_values = np.asarray(params, dtype=np.float64)
    probability_values = np.asarray(P, dtype=np.float64)
    return bool(
        n_samples >= 2
        and params_values.shape == (n_samples * n_components,)
        and probability_values.shape == (_condensed_length(n_samples),)
        and np.all(probability_values >= 0.0)
        and np.sum(probability_values) > 0.0
    )


def _kl_result_valid(result: TsneObjective, params: NDArray[np.float64], compute_error: bool) -> bool:
    if not (isinstance(result, tuple) and len(result) == 2):
        return False
    error, grad = result
    gradient = np.asarray(grad, dtype=np.float64)
    error_ok = bool(np.isfinite(float(error))) if compute_error else bool(np.isnan(float(error)))
    return bool(error_ok and gradient.shape == np.asarray(params).shape and np.all(np.isfinite(gradient)))


def _update_vectors_valid(
    p: NDArray[np.float64],
    update: NDArray[np.float64],
    gains: NDArray[np.float64],
    grad: NDArray[np.float64],
) -> bool:
    if not (_finite_vector(p) and _finite_vector(update) and _finite_vector(gains) and _finite_vector(grad)):
        return False
    shape = np.asarray(p).shape
    return bool(np.asarray(update).shape == shape and np.asarray(gains).shape == shape and np.asarray(grad).shape == shape and np.all(np.asarray(gains, dtype=np.float64) > 0.0))


def _update_result_valid(result: TsneUpdate, p: NDArray[np.float64]) -> bool:
    if not (isinstance(result, tuple) and len(result) == 3):
        return False
    new_p, new_update, new_gains = result
    target_shape = np.asarray(p).shape
    return bool(
        np.asarray(new_p).shape == target_shape
        and np.asarray(new_update).shape == target_shape
        and np.asarray(new_gains).shape == target_shape
        and np.all(np.isfinite(new_p))
        and np.all(np.isfinite(new_update))
        and np.all(np.isfinite(new_gains))
        and np.all(np.asarray(new_gains, dtype=np.float64) > 0.0)
    )


@register_atom(witness_tsne_exact_joint_probabilities)
@icontract.require(lambda conditional_probabilities: _nonnegative_square_matrix(conditional_probabilities), "conditional probabilities must be a finite nonnegative square matrix")
@icontract.require(lambda conditional_probabilities: np.sum(np.asarray(conditional_probabilities, dtype=np.float64)) > 0.0, "conditional probabilities must carry positive mass")
@icontract.ensure(lambda result, conditional_probabilities: _joint_result_valid(result, conditional_probabilities), "joint probabilities must be finite positive condensed probabilities")
def tsne_exact_joint_probabilities(conditional_probabilities: NDArray[np.float64]) -> NDArray[np.float64]:
    """Symmetrize and normalize dense t-SNE conditional probabilities."""
    conditional = np.asarray(conditional_probabilities, dtype=np.float64)
    joint = conditional + conditional.T
    sum_joint = np.maximum(np.sum(joint), MACHINE_EPSILON)
    return np.maximum(squareform(joint) / sum_joint, MACHINE_EPSILON)


@register_atom(witness_tsne_exact_kl_divergence)
@icontract.require(lambda params, P, degrees_of_freedom, n_samples, n_components, skip_num_points: _kl_inputs_valid(params, P, degrees_of_freedom, n_samples, n_components, skip_num_points), "exact t-SNE inputs must have compatible condensed probability and embedding shapes")
@icontract.ensure(lambda result, params, compute_error: _kl_result_valid(result, params, compute_error), "KL result must contain a scalar error and finite gradient")
def tsne_exact_kl_divergence(
    params: NDArray[np.float64],
    P: NDArray[np.float64],
    degrees_of_freedom: int,
    n_samples: int,
    n_components: int,
    *,
    skip_num_points: int = 0,
    compute_error: bool = True,
) -> TsneObjective:
    """Compute the exact dense t-SNE KL objective and gradient."""
    embedded = np.asarray(params, dtype=np.float64).reshape(n_samples, n_components)
    probabilities = np.asarray(P, dtype=np.float64)

    distances = pdist(embedded, "sqeuclidean")
    distances /= degrees_of_freedom
    distances += 1.0
    distances **= (degrees_of_freedom + 1.0) / -2.0
    Q = np.maximum(distances / (2.0 * np.sum(distances)), MACHINE_EPSILON)

    if compute_error:
        kl_divergence = float(2.0 * np.dot(probabilities, np.log(np.maximum(probabilities, MACHINE_EPSILON) / Q)))
    else:
        kl_divergence = float(np.nan)

    grad = np.zeros((n_samples, n_components), dtype=np.float64)
    weighted_distances = squareform((probabilities - Q) * distances)
    for i in range(skip_num_points, n_samples):
        grad[i] = np.dot(np.ravel(weighted_distances[i], order="K"), embedded[i] - embedded)
    grad = grad.ravel()
    grad *= 2.0 * (degrees_of_freedom + 1.0) / degrees_of_freedom
    return kl_divergence, grad


@register_atom(witness_tsne_gradient_descent_update)
@icontract.require(lambda p, update, gains, grad: _update_vectors_valid(p, update, gains, grad), "state, update, gains, and gradient must be finite equal-length vectors")
@icontract.require(lambda momentum: _unit_interval(momentum), "momentum must be finite in [0, 1)")
@icontract.require(lambda learning_rate: _positive_finite(learning_rate), "learning rate must be positive and finite")
@icontract.require(lambda min_gain: _positive_finite(min_gain), "minimum gain must be positive and finite")
@icontract.ensure(lambda result, p: _update_result_valid(result, p), "updated state, update, and gains must remain finite")
def tsne_gradient_descent_update(
    p: NDArray[np.float64],
    update: NDArray[np.float64],
    gains: NDArray[np.float64],
    grad: NDArray[np.float64],
    *,
    momentum: float = 0.8,
    learning_rate: float = 200.0,
    min_gain: float = 0.01,
) -> TsneUpdate:
    """Apply one t-SNE momentum and adaptive-gain gradient-descent update."""
    p_values = np.asarray(p, dtype=np.float64).copy().ravel()
    update_values = np.asarray(update, dtype=np.float64).copy().ravel()
    gain_values = np.asarray(gains, dtype=np.float64).copy().ravel()
    grad_values = np.asarray(grad, dtype=np.float64).copy().ravel()

    increasing = update_values * grad_values < 0.0
    decreasing = np.invert(increasing)
    gain_values[increasing] += 0.2
    gain_values[decreasing] *= 0.8
    np.clip(gain_values, min_gain, np.inf, out=gain_values)
    scaled_grad = grad_values * gain_values
    new_update = float(momentum) * update_values - float(learning_rate) * scaled_grad
    new_p = p_values + new_update
    return new_p, new_update, gain_values
