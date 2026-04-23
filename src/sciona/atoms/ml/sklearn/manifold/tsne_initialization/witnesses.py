"""Ghost witnesses for t-SNE starting-point and scheduling atoms."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_tsne_auto_learning_rate(n_samples: int, early_exaggeration: float) -> float:
    """Describe sklearn's auto learning-rate scalar."""
    if n_samples < 2:
        raise ValueError("n_samples must be at least 2")
    if early_exaggeration <= 0.0:
        raise ValueError("early_exaggeration must be positive")
    return 50.0


def witness_tsne_barnes_hut_neighbor_count(n_samples: int, perplexity: float) -> int:
    """Describe the Barnes-Hut nearest-neighbor count."""
    if n_samples < 2:
        raise ValueError("n_samples must be at least 2")
    if perplexity <= 0.0:
        raise ValueError("perplexity must be positive")
    return 1


def witness_tsne_random_initialize_embedding(
    n_samples: int,
    n_components: int,
    random_state: int | None = None,
) -> AbstractArray:
    """Describe an iid Gaussian t-SNE start matrix."""
    del random_state
    if n_samples < 1 or n_components < 1:
        raise ValueError("n_samples and n_components must be positive")
    return AbstractArray(shape=(n_samples, n_components), dtype="float32")


def witness_tsne_pca_rescale_embedding(embedding: AbstractArray) -> AbstractArray:
    """Describe PCA-rescaled t-SNE start coordinates."""
    if len(embedding.shape) != 2:
        raise ValueError("embedding must be 2D")
    n_samples, n_components = int(embedding.shape[0]), int(embedding.shape[1])
    if n_samples < 1 or n_components < 1:
        raise ValueError("embedding must be nonempty")
    return AbstractArray(shape=(n_samples, n_components), dtype="float32")


def witness_tsne_degrees_of_freedom(n_components: int) -> int:
    """Describe t-SNE Student-t degrees of freedom."""
    if n_components < 1:
        raise ValueError("n_components must be positive")
    return 1
