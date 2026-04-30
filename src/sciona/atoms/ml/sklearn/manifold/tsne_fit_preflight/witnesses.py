"""Ghost witnesses for t-SNE fit preflight helpers."""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray


def witness_tsne_fit_require_perplexity_below_sample_count(
    perplexity: float,
    *,
    n_samples: int,
) -> bool:
    """Describe sklearn's perplexity-versus-sample-count fit preflight."""
    del perplexity
    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    return True


def witness_tsne_fit_accept_sparse_formats(method: str) -> AbstractArray:
    """Describe the accepted sparse storage formats selected from the t-SNE method."""
    if not isinstance(method, str) or len(method) == 0:
        raise ValueError("method must be a nonempty string")
    return AbstractArray(shape=(None,), dtype="str")


def witness_tsne_fit_require_sparse_input_init_not_pca(
    init: object,
    *,
    is_sparse_input: bool,
) -> bool:
    """Describe the sparse-input guard that rejects PCA initialization."""
    del init, is_sparse_input
    return True


def witness_tsne_fit_require_precomputed_square_matrix(shape: tuple[int, int]) -> bool:
    """Describe the precomputed-distance square-matrix shape guard."""
    del shape
    return True


def witness_tsne_fit_require_dense_exact_precomputed(
    method: str,
    metric: object,
    *,
    is_sparse_input: bool,
) -> bool:
    """Describe the exact-method guard against sparse precomputed distances."""
    del method, metric, is_sparse_input
    return True


def witness_tsne_fit_require_barnes_hut_components(
    method: str,
    *,
    n_components: int,
) -> bool:
    """Describe the Barnes-Hut dimensionality guard in t-SNE fit."""
    del method
    if n_components < 1:
        raise ValueError("n_components must be positive")
    return True
