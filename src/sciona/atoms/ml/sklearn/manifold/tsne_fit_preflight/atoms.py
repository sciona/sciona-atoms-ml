"""t-SNE fit preflight atoms adapted from scikit-learn."""

from __future__ import annotations

import icontract
import numpy as np

from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_tsne_fit_accept_sparse_formats,
    witness_tsne_fit_require_barnes_hut_components,
    witness_tsne_fit_require_dense_exact_precomputed,
    witness_tsne_fit_require_perplexity_below_sample_count,
    witness_tsne_fit_require_precomputed_square_matrix,
    witness_tsne_fit_require_sparse_input_init_not_pca,
)


def _method_valid(method: str) -> bool:
    return method in {"barnes_hut", "exact"}


def _shape_valid(shape: tuple[int, int]) -> bool:
    return bool(
        isinstance(shape, tuple)
        and len(shape) == 2
        and all(isinstance(dim, int) and dim >= 1 for dim in shape)
    )


def _positive_n_samples(n_samples: int) -> bool:
    return isinstance(n_samples, int) and not isinstance(n_samples, bool) and n_samples >= 1


def _positive_n_components(n_components: int) -> bool:
    return isinstance(n_components, int) and not isinstance(n_components, bool) and n_components >= 1


@register_atom(witness_tsne_fit_require_perplexity_below_sample_count)
@icontract.require(
    lambda perplexity: isinstance(perplexity, (int, float)) and not isinstance(perplexity, bool) and np.isfinite(float(perplexity)),
    "perplexity must be finite",
)
@icontract.require(lambda n_samples: _positive_n_samples(n_samples), "n_samples must be a positive integer")
@icontract.require(
    lambda perplexity, n_samples: float(perplexity) < n_samples,
    "perplexity must be less than n_samples",
)
@icontract.ensure(lambda result: result is True, "successful preflight returns True")
def tsne_fit_require_perplexity_below_sample_count(
    perplexity: float,
    *,
    n_samples: int,
) -> bool:
    """Require sklearn's perplexity-versus-sample-count fit preflight rule."""
    return True


@register_atom(witness_tsne_fit_accept_sparse_formats)
@icontract.require(lambda method: _method_valid(method), "method must be 'barnes_hut' or 'exact'")
@icontract.ensure(
    lambda method, result: result == (("csr",) if method == "barnes_hut" else ("csr", "csc", "coo")),
    "result must match sklearn's sparse-format acceptance by method",
)
def tsne_fit_accept_sparse_formats(method: str) -> tuple[str, ...]:
    """Return sklearn's accepted sparse formats for TSNE._fit validation."""
    if method == "barnes_hut":
        return ("csr",)
    return ("csr", "csc", "coo")


@register_atom(witness_tsne_fit_require_sparse_input_init_not_pca)
@icontract.require(lambda is_sparse_input: isinstance(is_sparse_input, bool), "is_sparse_input must be boolean")
@icontract.require(
    lambda init, is_sparse_input: not (is_sparse_input and isinstance(init, str) and init == "pca"),
    "sparse input does not support init='pca'",
)
@icontract.ensure(lambda result: result is True, "successful preflight returns True")
def tsne_fit_require_sparse_input_init_not_pca(
    init: object,
    *,
    is_sparse_input: bool,
) -> bool:
    """Require sklearn's sparse-input guard against PCA initialization."""
    return True


@register_atom(witness_tsne_fit_require_precomputed_square_matrix)
@icontract.require(lambda shape: _shape_valid(shape), "shape must be a two-dimensional positive integer tuple")
@icontract.require(lambda shape: shape[0] == shape[1], "precomputed distance matrices must be square")
@icontract.ensure(lambda result: result is True, "successful preflight returns True")
def tsne_fit_require_precomputed_square_matrix(shape: tuple[int, int]) -> bool:
    """Require sklearn's square-matrix guard for precomputed distances."""
    return True


@register_atom(witness_tsne_fit_require_dense_exact_precomputed)
@icontract.require(lambda method: _method_valid(method), "method must be 'barnes_hut' or 'exact'")
@icontract.require(lambda is_sparse_input: isinstance(is_sparse_input, bool), "is_sparse_input must be boolean")
@icontract.require(
    lambda method, metric, is_sparse_input: not (method == "exact" and metric == "precomputed" and is_sparse_input),
    "exact t-SNE does not accept sparse precomputed distances",
)
@icontract.ensure(lambda result: result is True, "successful preflight returns True")
def tsne_fit_require_dense_exact_precomputed(
    method: str,
    metric: object,
    *,
    is_sparse_input: bool,
) -> bool:
    """Require sklearn's exact-method guard against sparse precomputed distances."""
    return True


@register_atom(witness_tsne_fit_require_barnes_hut_components)
@icontract.require(lambda method: _method_valid(method), "method must be 'barnes_hut' or 'exact'")
@icontract.require(lambda n_components: _positive_n_components(n_components), "n_components must be a positive integer")
@icontract.require(
    lambda method, n_components: method != "barnes_hut" or n_components <= 3,
    "barnes_hut t-SNE requires n_components <= 3",
)
@icontract.ensure(lambda result: result is True, "successful preflight returns True")
def tsne_fit_require_barnes_hut_components(
    method: str,
    *,
    n_components: int,
) -> bool:
    """Require sklearn's Barnes-Hut dimensionality preflight rule."""
    return True
