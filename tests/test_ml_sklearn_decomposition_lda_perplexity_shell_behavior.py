from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError
from sklearn.decomposition import LatentDirichletAllocation


def _tiny_counts() -> np.ndarray:
    return np.array(
        [
            [1.0, 2.0, 0.0, 1.0],
            [0.0, 1.0, 3.0, 0.0],
            [2.0, 0.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )


def test_lda_perplexity_shell_atoms_import() -> None:
    from sciona.atoms.ml.sklearn.decomposition.lda_perplexity_shell import (
        lda_fit_transform_output,
        lda_perplexity_precomputed_topics,
    )

    assert callable(lda_perplexity_precomputed_topics)
    assert callable(lda_fit_transform_output)


def test_lda_perplexity_shell_matches_sklearn_shell() -> None:
    from sciona.atoms.ml.sklearn.decomposition.lda_perplexity_shell import (
        lda_fit_transform_output,
        lda_perplexity_precomputed_topics,
    )

    X = _tiny_counts()
    model = LatentDirichletAllocation(n_components=2, random_state=0, max_iter=1, total_samples=10.0)
    model.fit(X)

    transformed = model.fit_transform(X, normalize=True)
    assert np.allclose(lda_fit_transform_output(transformed), transformed)

    doc_topic = model._unnormalized_transform(X)
    bound = model._approx_bound(X, doc_topic, sub_sampling=False)
    assert lda_perplexity_precomputed_topics(bound, X, doc_topic, n_components=model.n_components) == pytest.approx(
        model._perplexity_precomp_distr(X, doc_topic, sub_sampling=False)
    )

    bound_sub = model._approx_bound(X, doc_topic, sub_sampling=True)
    assert lda_perplexity_precomputed_topics(
        bound_sub,
        X,
        doc_topic,
        n_components=model.n_components,
        total_samples=model.total_samples,
        sub_sampling=True,
    ) == pytest.approx(model._perplexity_precomp_distr(X, doc_topic, sub_sampling=True))


def test_lda_perplexity_shell_matches_sklearn_errors() -> None:
    from sciona.atoms.ml.sklearn.decomposition.lda_perplexity_shell import lda_perplexity_precomputed_topics

    X = _tiny_counts()
    topics = np.ones((2, 2), dtype=np.float64)
    with pytest.raises(ValueError, match="Number of samples in X and doc_topic_distr do not match."):
        lda_perplexity_precomputed_topics(1.0, X, topics, n_components=2)

    topics = np.ones((3, 3), dtype=np.float64)
    with pytest.raises(ValueError, match="Number of topics does not match."):
        lda_perplexity_precomputed_topics(1.0, X, topics, n_components=2)


def test_lda_perplexity_shell_contracts_reject_invalid_inputs() -> None:
    from sciona.atoms.ml.sklearn.decomposition.lda_perplexity_shell import (
        lda_fit_transform_output,
        lda_perplexity_precomputed_topics,
    )

    with pytest.raises(ViolationError):
        lda_fit_transform_output(np.array([[1.0, 0.0]], dtype=np.float64))

    with pytest.raises(ViolationError):
        lda_perplexity_precomputed_topics(float("nan"), np.ones((2, 2)), np.ones((2, 2)), n_components=2)
