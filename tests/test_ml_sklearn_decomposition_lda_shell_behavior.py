from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError
from sklearn.decomposition import LatentDirichletAllocation

from sciona.atoms.ml.sklearn.decomposition.lda_shell import (
    lda_doc_topic_prior,
    lda_component_values,
    lda_normalize_document_topics,
    lda_perplexity_from_bound,
    lda_perplexity_require_matching_samples,
    lda_perplexity_require_matching_topics,
    lda_perplexity_word_count,
    lda_topic_word_prior,
)


def _tiny_counts() -> np.ndarray:
    return np.array(
        [
            [1.0, 2.0, 0.0, 1.0],
            [0.0, 1.0, 3.0, 0.0],
            [2.0, 0.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )


def test_lda_shell_atoms_import() -> None:
    assert callable(lda_doc_topic_prior)
    assert callable(lda_topic_word_prior)
    assert callable(lda_component_values)
    assert callable(lda_normalize_document_topics)
    assert callable(lda_perplexity_require_matching_samples)
    assert callable(lda_perplexity_require_matching_topics)
    assert callable(lda_perplexity_word_count)
    assert callable(lda_perplexity_from_bound)


def test_lda_prior_resolution_and_initial_components_match_private_init() -> None:
    model = LatentDirichletAllocation(
        n_components=3,
        random_state=7,
        doc_topic_prior=None,
        topic_word_prior=0.25,
    )
    model._init_latent_vars(n_features=4, dtype=np.float32)

    assert lda_doc_topic_prior(3, doc_topic_prior=None) == model.doc_topic_prior_
    assert lda_topic_word_prior(3, topic_word_prior=0.25) == model.topic_word_prior_

    observed = lda_component_values(3, 4, random_state=7, dtype_name="float32")
    assert observed.dtype == np.float32
    assert np.allclose(observed, model.components_)


def test_lda_normalize_document_topics_matches_transform_normalization_rule() -> None:
    doc_topic = np.array([[2.0, 1.0, 1.0], [3.0, 3.0, 6.0]], dtype=np.float64)

    observed = lda_normalize_document_topics(doc_topic)
    expected = doc_topic / doc_topic.sum(axis=1)[:, np.newaxis]

    assert np.allclose(observed, expected)
    assert np.allclose(observed.sum(axis=1), 1.0)


def test_lda_perplexity_validators_match_sklearn_errors() -> None:
    X = _tiny_counts()
    model = LatentDirichletAllocation(n_components=2, random_state=0, max_iter=1)
    model.fit(X)

    with pytest.raises(ValueError, match="Number of samples in X and doc_topic_distr do not match."):
        model._perplexity_precomp_distr(X, np.ones((2, 2), dtype=np.float64))
    with pytest.raises(ValueError, match="Number of samples in X and doc_topic_distr do not match."):
        lda_perplexity_require_matching_samples(np.ones((2, 2), dtype=np.float64), n_samples=X.shape[0])

    valid_samples = lda_perplexity_require_matching_samples(
        np.ones((X.shape[0], 3), dtype=np.float64),
        n_samples=X.shape[0],
    )
    with pytest.raises(ValueError, match="Number of topics does not match."):
        model._perplexity_precomp_distr(X, np.ones((X.shape[0], 3), dtype=np.float64))
    with pytest.raises(ValueError, match="Number of topics does not match."):
        lda_perplexity_require_matching_topics(valid_samples, n_components=model.n_components)


def test_lda_perplexity_math_matches_private_helper() -> None:
    X = _tiny_counts()
    model = LatentDirichletAllocation(n_components=2, random_state=3, max_iter=1, total_samples=10.0)
    model.fit(X)

    doc_topic_distr = model._unnormalized_transform(X)
    bound = model._approx_bound(X, doc_topic_distr, sub_sampling=False)
    word_count = lda_perplexity_word_count(
        float(X.sum()),
        current_samples=X.shape[0],
        total_samples=model.total_samples,
        sub_sampling=False,
    )
    observed = lda_perplexity_from_bound(bound, word_count=word_count)
    expected = model._perplexity_precomp_distr(X, doc_topic_distr, sub_sampling=False)

    assert np.allclose(observed, expected)

    sub_word_count = lda_perplexity_word_count(
        float(X.sum()),
        current_samples=X.shape[0],
        total_samples=model.total_samples,
        sub_sampling=True,
    )
    sub_bound = model._approx_bound(X, doc_topic_distr, sub_sampling=True)
    observed_sub = lda_perplexity_from_bound(sub_bound, word_count=sub_word_count)
    expected_sub = model._perplexity_precomp_distr(X, doc_topic_distr, sub_sampling=True)
    assert np.allclose(observed_sub, expected_sub)


def test_lda_shell_contracts_reject_invalid_inputs() -> None:
    with pytest.raises(ViolationError):
        lda_doc_topic_prior(0, doc_topic_prior=None)

    with pytest.raises(ViolationError):
        lda_component_values(2, 0, random_state=0)

    with pytest.raises(ViolationError):
        lda_normalize_document_topics(np.array([[0.0, 0.0]], dtype=np.float64))

    with pytest.raises(ViolationError):
        lda_perplexity_word_count(0.0, current_samples=2, total_samples=10.0)
