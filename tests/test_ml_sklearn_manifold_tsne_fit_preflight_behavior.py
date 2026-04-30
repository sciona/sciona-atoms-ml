from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError
from scipy.sparse import csr_matrix
from sklearn.manifold import TSNE


def test_tsne_fit_preflight_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.manifold.tsne_fit_preflight import (
        tsne_fit_accept_sparse_formats,
        tsne_fit_require_barnes_hut_components,
        tsne_fit_require_dense_exact_precomputed,
        tsne_fit_require_perplexity_below_sample_count,
        tsne_fit_require_precomputed_square_matrix,
        tsne_fit_require_sparse_input_init_not_pca,
    )

    assert callable(tsne_fit_require_perplexity_below_sample_count)
    assert callable(tsne_fit_accept_sparse_formats)
    assert callable(tsne_fit_require_sparse_input_init_not_pca)
    assert callable(tsne_fit_require_precomputed_square_matrix)
    assert callable(tsne_fit_require_dense_exact_precomputed)
    assert callable(tsne_fit_require_barnes_hut_components)


def test_tsne_fit_accept_sparse_formats_matches_sklearn_policy() -> None:
    from sciona.atoms.ml.sklearn.manifold.tsne_fit_preflight import tsne_fit_accept_sparse_formats

    assert tsne_fit_accept_sparse_formats("barnes_hut") == ("csr",)
    assert tsne_fit_accept_sparse_formats("exact") == ("csr", "csc", "coo")


def test_tsne_fit_preflight_valid_cases_match_sklearn_shell() -> None:
    from sciona.atoms.ml.sklearn.manifold.tsne_fit_preflight import (
        tsne_fit_require_barnes_hut_components,
        tsne_fit_require_dense_exact_precomputed,
        tsne_fit_require_perplexity_below_sample_count,
        tsne_fit_require_precomputed_square_matrix,
        tsne_fit_require_sparse_input_init_not_pca,
    )

    assert tsne_fit_require_perplexity_below_sample_count(2.5, n_samples=10) is True
    assert tsne_fit_require_sparse_input_init_not_pca("random", is_sparse_input=True) is True
    assert tsne_fit_require_sparse_input_init_not_pca(np.zeros((2, 2), dtype=float), is_sparse_input=True) is True
    assert tsne_fit_require_precomputed_square_matrix((3, 3)) is True
    assert tsne_fit_require_dense_exact_precomputed("exact", "euclidean", is_sparse_input=True) is True
    assert tsne_fit_require_dense_exact_precomputed("barnes_hut", "precomputed", is_sparse_input=True) is True
    assert tsne_fit_require_barnes_hut_components("barnes_hut", n_components=3) is True
    assert tsne_fit_require_barnes_hut_components("exact", n_components=5) is True


def test_tsne_fit_preflight_invalid_cases_match_sklearn_raises() -> None:
    from sciona.atoms.ml.sklearn.manifold.tsne_fit_preflight import (
        tsne_fit_require_barnes_hut_components,
        tsne_fit_require_dense_exact_precomputed,
        tsne_fit_require_perplexity_below_sample_count,
        tsne_fit_require_precomputed_square_matrix,
        tsne_fit_require_sparse_input_init_not_pca,
    )

    with pytest.raises(ValueError, match="perplexity must be less than n_samples"):
        TSNE(perplexity=2).fit_transform(np.array([[0.0], [1.0]], dtype=float))
    with pytest.raises(ViolationError):
        tsne_fit_require_perplexity_below_sample_count(2, n_samples=2)

    sparse_samples = csr_matrix([[0.0, 1.0], [1.0, 0.0]], dtype=float)
    with pytest.raises(TypeError, match='PCA initialization is currently not supported with the sparse input matrix'):
        TSNE(init="pca", perplexity=1).fit_transform(sparse_samples)
    with pytest.raises(ViolationError):
        tsne_fit_require_sparse_input_init_not_pca("pca", is_sparse_input=True)

    precomputed = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float)
    nonsquare = np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 3.0]], dtype=float)
    with pytest.raises(ValueError, match="X should be a square distance matrix"):
        TSNE(metric="precomputed", init="random", perplexity=1).fit_transform(nonsquare)
    with pytest.raises(ViolationError):
        tsne_fit_require_precomputed_square_matrix((2, 3))

    sparse_precomputed = csr_matrix(precomputed)
    with pytest.raises(TypeError, match='TSNE with method="exact" does not accept sparse precomputed distance matrix'):
        TSNE(method="exact", metric="precomputed", init="random", perplexity=1).fit_transform(sparse_precomputed)
    with pytest.raises(ViolationError):
        tsne_fit_require_dense_exact_precomputed("exact", "precomputed", is_sparse_input=True)

    with pytest.raises(ValueError, match="'n_components' should be inferior to 4 for the barnes_hut algorithm"):
        TSNE(method="barnes_hut", n_components=4, perplexity=1).fit_transform(np.array([[0.0], [1.0]], dtype=float))
    with pytest.raises(ViolationError):
        tsne_fit_require_barnes_hut_components("barnes_hut", n_components=4)


def test_tsne_fit_preflight_contracts_reject_invalid_inputs() -> None:
    from sciona.atoms.ml.sklearn.manifold.tsne_fit_preflight import (
        tsne_fit_accept_sparse_formats,
        tsne_fit_require_barnes_hut_components,
        tsne_fit_require_perplexity_below_sample_count,
        tsne_fit_require_precomputed_square_matrix,
    )

    with pytest.raises(ViolationError):
        tsne_fit_accept_sparse_formats("approximate")

    with pytest.raises(ViolationError):
        tsne_fit_require_perplexity_below_sample_count(np.inf, n_samples=5)

    with pytest.raises(ViolationError):
        tsne_fit_require_precomputed_square_matrix((0, 0))

    with pytest.raises(ViolationError):
        tsne_fit_require_barnes_hut_components("exact", n_components=0)
