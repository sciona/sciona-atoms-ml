"""t-SNE starting-point and scheduling atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_tsne_auto_learning_rate,
    witness_tsne_barnes_hut_neighbor_count,
    witness_tsne_degrees_of_freedom,
    witness_tsne_pca_rescale_embedding,
    witness_tsne_random_initialize_embedding,
)

def _positive_int(value: int) -> bool:
    return bool(isinstance(value, int) and not isinstance(value, bool) and value >= 1)

def _n_samples_valid(value: int) -> bool:
    return bool(_positive_int(value) and value >= 2)

def _positive_finite(value: float) -> bool:
    return bool(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and np.isfinite(float(value))
        and float(value) > 0.0
    )

def _random_state_valid(value: int | None) -> bool:
    return value is None or (isinstance(value, int) and not isinstance(value, bool) and value >= 0)

def _finite_matrix(values: NDArray[np.floating]) -> bool:
    try:
        array = np.asarray(values, dtype=np.float32)
    except (TypeError, ValueError):
        return False
    return bool(array.ndim == 2 and array.shape[0] >= 1 and array.shape[1] >= 1 and np.all(np.isfinite(array)))

def _nonzero_first_column_std(values: NDArray[np.floating]) -> bool:
    if not _finite_matrix(values):
        return False
    array = np.asarray(values, dtype=np.float32)
    return bool(np.std(array[:, 0]) > 0.0)

def _float32_matrix_shape(result: NDArray[np.float32], n_samples: int, n_components: int) -> bool:
    values = np.asarray(result)
    return bool(values.shape == (n_samples, n_components) and values.dtype == np.float32 and np.all(np.isfinite(values)))

def _same_shape_float32(result: NDArray[np.float32], embedding: NDArray[np.floating]) -> bool:
    values = np.asarray(result)
    source = np.asarray(embedding)
    return bool(values.shape == source.shape and values.dtype == np.float32 and np.all(np.isfinite(values)))

@register_atom(witness_tsne_auto_learning_rate)
@icontract.require(lambda n_samples: _n_samples_valid(n_samples), "n_samples must be an integer >= 2")
@icontract.require(lambda early_exaggeration: _positive_finite(early_exaggeration), "early_exaggeration must be positive and finite")
@icontract.ensure(lambda result: _positive_finite(result) and float(result) >= 50.0, "auto learning rate must be finite, positive, and at least 50")
def tsne_auto_learning_rate(n_samples: int, early_exaggeration: float) -> float:
    """Compute sklearn's automatic t-SNE learning rate."""
    return float(np.maximum(n_samples / float(early_exaggeration) / 4.0, 50.0))

@register_atom(witness_tsne_barnes_hut_neighbor_count)
@icontract.require(lambda n_samples: _n_samples_valid(n_samples), "n_samples must be an integer >= 2")
@icontract.require(lambda perplexity: _positive_finite(perplexity), "perplexity must be positive and finite")
@icontract.ensure(lambda result, n_samples: _positive_int(result) and result <= n_samples - 1, "neighbor count must be a positive integer not exceeding n_samples - 1")
def tsne_barnes_hut_neighbor_count(n_samples: int, perplexity: float) -> int:
    """Compute sklearn's Barnes-Hut t-SNE nearest-neighbor count."""
    return int(min(n_samples - 1, int(3.0 * float(perplexity) + 1)))

@register_atom(witness_tsne_random_initialize_embedding)
@icontract.require(lambda n_samples: _positive_int(n_samples), "n_samples must be a positive integer")
@icontract.require(lambda n_components: _positive_int(n_components), "n_components must be a positive integer")
@icontract.require(lambda random_state: _random_state_valid(random_state), "random_state must be None or a nonnegative integer")
@icontract.ensure(lambda result, n_samples, n_components: _float32_matrix_shape(result, n_samples, n_components), "random start matrix must be a finite float32 matrix with the requested shape")
def tsne_random_initialize_embedding(
    n_samples: int,
    n_components: int,
    *,
    random_state: int | None = None,
) -> NDArray[np.float32]:
    from sklearn.utils import check_random_state
    """Draw a t-SNE start matrix from iid Gaussian samples scaled by 1e-4."""
    rng = check_random_state(random_state)
    return np.asarray(
        1e-4 * rng.standard_normal(size=(n_samples, n_components)).astype(np.float32),
        dtype=np.float32,
    )

@register_atom(witness_tsne_pca_rescale_embedding)
@icontract.require(lambda embedding: _nonzero_first_column_std(embedding), "embedding must be a finite 2D matrix whose first column has nonzero standard deviation")
@icontract.ensure(lambda result, embedding: _same_shape_float32(result, embedding), "rescaled PCA embedding must preserve shape and use float32")
def tsne_pca_rescale_embedding(embedding: NDArray[np.floating]) -> NDArray[np.float32]:
    """Rescale a PCA embedding so its first component has standard deviation 1e-4."""
    values = np.asarray(embedding, dtype=np.float32)
    return np.asarray(values / np.std(values[:, 0]) * 1e-4, dtype=np.float32)

@register_atom(witness_tsne_degrees_of_freedom)
@icontract.require(lambda n_components: _positive_int(n_components), "n_components must be a positive integer")
@icontract.ensure(lambda result: _positive_int(result), "degrees of freedom must be a positive integer")
def tsne_degrees_of_freedom(n_components: int) -> int:
    """Compute the Student-t degrees of freedom sklearn uses for t-SNE."""
    return int(max(n_components - 1, 1))
