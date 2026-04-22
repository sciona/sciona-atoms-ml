"""Neighborhood Components Analysis helper atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_nca_linear_transform,
    witness_nca_loss_gradient,
    witness_nca_neighbor_probabilities,
    witness_nca_same_class_mask,
)


def _finite_matrix(values: NDArray[np.float64]) -> bool:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 2 and array.shape[0] >= 2 and array.shape[1] >= 1 and np.all(np.isfinite(array)))


def _integer_label_vector(y: NDArray[np.int64]) -> bool:
    values = np.asarray(y)
    return bool(values.ndim == 1 and values.shape[0] >= 2 and np.issubdtype(values.dtype, np.integer))


def _feature_counts_match(X: NDArray[np.float64], components: NDArray[np.float64]) -> bool:
    values = np.asarray(X)
    component_values = np.asarray(components)
    return bool(values.ndim == 2 and component_values.ndim == 2 and values.shape[1] == component_values.shape[1])


def _bool_square_mask(mask: NDArray[np.bool_], n_samples: int) -> bool:
    values = np.asarray(mask)
    return bool(values.ndim == 2 and values.shape == (n_samples, n_samples) and values.dtype == np.bool_)


def _transformation_valid(transformation: NDArray[np.float64], X: NDArray[np.float64]) -> bool:
    try:
        values = np.asarray(transformation, dtype=np.float64)
        samples = np.asarray(X, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return bool(
        values.ndim == 1
        and samples.ndim == 2
        and samples.shape[1] >= 1
        and values.shape[0] >= samples.shape[1]
        and values.shape[0] % samples.shape[1] == 0
        and np.all(np.isfinite(values))
    )


def _sign_valid(sign: float) -> bool:
    return bool(isinstance(sign, (int, float)) and not isinstance(sign, bool) and np.isfinite(float(sign)))


def _same_class_result_valid(result: NDArray[np.bool_], y: NDArray[np.int64]) -> bool:
    values = np.asarray(result)
    labels = np.asarray(y)
    return bool(values.dtype == np.bool_ and values.shape == (labels.shape[0], labels.shape[0]) and np.all(np.diag(values)))


def _linear_transform_result_valid(
    result: NDArray[np.float64],
    X: NDArray[np.float64],
    components: NDArray[np.float64],
) -> bool:
    values = np.asarray(result, dtype=np.float64)
    samples = np.asarray(X)
    component_values = np.asarray(components)
    return bool(values.shape == (samples.shape[0], component_values.shape[0]) and np.all(np.isfinite(values)))


def _probability_result_valid(result: NDArray[np.float64], X_embedded: NDArray[np.float64]) -> bool:
    values = np.asarray(result, dtype=np.float64)
    embedded = np.asarray(X_embedded)
    return bool(
        values.shape == (embedded.shape[0], embedded.shape[0])
        and np.all(np.isfinite(values))
        and np.all(values >= 0.0)
        and np.allclose(values.sum(axis=1), 1.0)
        and np.allclose(np.diag(values), 0.0)
    )


def _loss_gradient_result_valid(
    result: tuple[float, NDArray[np.float64]],
    transformation: NDArray[np.float64],
) -> bool:
    loss, gradient = result
    gradient_values = np.asarray(gradient, dtype=np.float64)
    transform_values = np.asarray(transformation)
    return bool(
        isinstance(loss, float)
        and np.isfinite(loss)
        and gradient_values.shape == transform_values.shape
        and np.all(np.isfinite(gradient_values))
    )


@register_atom(witness_nca_same_class_mask)
@icontract.require(lambda y: _integer_label_vector(y), "y must be an encoded integer label vector")
@icontract.ensure(lambda result, y: _same_class_result_valid(result, y), "same-class mask must be square over labels")
def nca_same_class_mask(y: NDArray[np.int64]) -> NDArray[np.bool_]:
    """Build the fixed same-class mask used by NCA optimization."""
    labels = np.asarray(y, dtype=np.int64)
    return np.asarray(labels[:, np.newaxis] == labels[np.newaxis, :], dtype=np.bool_)


@register_atom(witness_nca_linear_transform)
@icontract.require(lambda X: _finite_matrix(X), "X must be a finite 2D matrix")
@icontract.require(lambda components: _finite_matrix(components), "components must be a finite 2D matrix")
@icontract.require(lambda X, components: _feature_counts_match(X, components), "component feature count must match X")
@icontract.ensure(lambda result, X, components: _linear_transform_result_valid(result, X, components), "transformed matrix must match sample and component counts")
def nca_linear_transform(
    X: NDArray[np.float64],
    components: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Apply an NCA component matrix to dense samples."""
    samples = np.asarray(X, dtype=np.float64)
    component_values = np.asarray(components, dtype=np.float64)
    return np.asarray(samples.dot(component_values.T), dtype=np.float64)


@register_atom(witness_nca_neighbor_probabilities)
@icontract.require(lambda X_embedded: _finite_matrix(X_embedded), "X_embedded must be a finite 2D matrix")
@icontract.ensure(lambda result, X_embedded: _probability_result_valid(result, X_embedded), "neighbor probabilities must be row-stochastic with zero diagonal")
def nca_neighbor_probabilities(X_embedded: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute how likely each transformed sample is to choose every other sample."""
    embedded = np.asarray(X_embedded, dtype=np.float64)
    row_norms = np.sum(embedded * embedded, axis=1, keepdims=True)
    distances = row_norms + row_norms.T - 2.0 * embedded.dot(embedded.T)
    distances = np.maximum(distances, 0.0)
    np.fill_diagonal(distances, np.inf)
    logits = -distances
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    weights = np.exp(shifted)
    weights[~np.isfinite(weights)] = 0.0
    return np.asarray(weights / weights.sum(axis=1, keepdims=True), dtype=np.float64)


@register_atom(witness_nca_loss_gradient)
@icontract.require(lambda X: _finite_matrix(X), "X must be a finite 2D matrix")
@icontract.require(lambda transformation, X: _transformation_valid(transformation, X), "transformation must flatten a finite component matrix over X features")
@icontract.require(lambda X, same_class_mask: _bool_square_mask(same_class_mask, np.asarray(X).shape[0]), "same_class_mask must be a boolean square mask over samples")
@icontract.require(lambda sign: _sign_valid(sign), "sign must be finite")
@icontract.ensure(lambda result, transformation: _loss_gradient_result_valid(result, transformation), "loss and flattened gradient must be finite")
def nca_loss_gradient(
    transformation: NDArray[np.float64],
    X: NDArray[np.float64],
    same_class_mask: NDArray[np.bool_],
    *,
    sign: float = 1.0,
) -> tuple[float, NDArray[np.float64]]:
    """Compute the NCA objective and flattened gradient for a transformation."""
    samples = np.asarray(X, dtype=np.float64)
    components = np.asarray(transformation, dtype=np.float64).reshape(-1, samples.shape[1])
    class_mask = np.asarray(same_class_mask, dtype=np.bool_)

    embedded = nca_linear_transform(samples, components)
    probabilities = nca_neighbor_probabilities(embedded)
    masked_probabilities = probabilities * class_mask
    per_sample_probability = np.sum(masked_probabilities, axis=1, keepdims=True)
    loss = float(np.sum(per_sample_probability))

    weighted_probabilities = masked_probabilities - probabilities * per_sample_probability
    symmetric_weights = weighted_probabilities + weighted_probabilities.T
    np.fill_diagonal(symmetric_weights, -weighted_probabilities.sum(axis=0))
    gradient = 2.0 * embedded.T.dot(symmetric_weights).dot(samples)
    return float(sign) * loss, np.asarray(float(sign) * gradient.ravel(), dtype=np.float64)
