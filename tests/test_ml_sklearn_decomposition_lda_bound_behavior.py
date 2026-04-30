from __future__ import annotations

import numpy as np
import pytest
from icontract import ViolationError
from scipy.special import gammaln, logsumexp
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.decomposition._lda import _dirichlet_expectation_2d

from sciona.atoms.ml.sklearn.decomposition.lda_bound import (
    lda_apply_subsampling_ratio,
    lda_approx_bound_from_expectations,
    lda_dirichlet_loglikelihood,
    lda_document_log_probability_bound,
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


def test_lda_bound_atoms_import() -> None:
    assert callable(lda_dirichlet_loglikelihood)
    assert callable(lda_document_log_probability_bound)
    assert callable(lda_apply_subsampling_ratio)
    assert callable(lda_approx_bound_from_expectations)


def test_lda_dirichlet_loglikelihood_matches_private_nested_helper() -> None:
    model = LatentDirichletAllocation(n_components=2, random_state=0, max_iter=1)
    X = _tiny_counts()
    model.fit(X)

    distr = model.components_
    dirichlet_distr = _dirichlet_expectation_2d(distr)
    prior = model.topic_word_prior_
    size = distr.shape[1]

    expected = np.sum((prior - distr) * dirichlet_distr)
    expected += np.sum(gammaln(distr) - gammaln(prior))
    expected += np.sum(gammaln(prior * size) - gammaln(np.sum(distr, axis=1)))

    observed = lda_dirichlet_loglikelihood(prior, distr, dirichlet_distr, size)
    assert np.allclose(observed, expected)


def test_lda_document_log_probability_bound_matches_private_bound_term() -> None:
    model = LatentDirichletAllocation(n_components=2, random_state=3, max_iter=1)
    X = _tiny_counts()
    model.fit(X)

    doc_topic = model._unnormalized_transform(X)
    dirichlet_doc_topic = _dirichlet_expectation_2d(doc_topic)
    dirichlet_component = _dirichlet_expectation_2d(model.components_)

    observed = lda_document_log_probability_bound(
        X,
        dirichlet_doc_topic,
        dirichlet_component,
    )

    expected = 0.0
    for idx_d in range(X.shape[0]):
        ids = np.nonzero(X[idx_d, :])[0]
        cnts = X[idx_d, ids]
        temp = dirichlet_doc_topic[idx_d, :, np.newaxis] + dirichlet_component[:, ids]
        expected += float(np.dot(cnts, logsumexp(temp, axis=0)))

    assert np.allclose(observed, expected)


def test_lda_apply_subsampling_ratio_matches_private_scaling() -> None:
    score = 12.5
    assert lda_apply_subsampling_ratio(
        score,
        total_samples=15.0,
        current_samples=3,
        sub_sampling=False,
    ) == pytest.approx(score)
    assert lda_apply_subsampling_ratio(
        score,
        total_samples=15.0,
        current_samples=3,
        sub_sampling=True,
    ) == pytest.approx(score * 5.0)


def test_lda_approx_bound_from_expectations_matches_private_method() -> None:
    X = _tiny_counts()
    model = LatentDirichletAllocation(
        n_components=2,
        random_state=4,
        max_iter=1,
        total_samples=10.0,
    )
    model.fit(X)

    doc_topic = model._unnormalized_transform(X)
    dirichlet_doc_topic = _dirichlet_expectation_2d(doc_topic)
    dirichlet_component = _dirichlet_expectation_2d(model.components_)

    observed = lda_approx_bound_from_expectations(
        X,
        doc_topic,
        dirichlet_doc_topic,
        model.components_,
        dirichlet_component,
        doc_topic_prior=model.doc_topic_prior_,
        topic_word_prior=model.topic_word_prior_,
        total_samples=model.total_samples,
        sub_sampling=False,
    )
    expected = model._approx_bound(X, doc_topic, sub_sampling=False)
    assert np.allclose(observed, expected)

    observed_sub = lda_approx_bound_from_expectations(
        X,
        doc_topic,
        dirichlet_doc_topic,
        model.components_,
        dirichlet_component,
        doc_topic_prior=model.doc_topic_prior_,
        topic_word_prior=model.topic_word_prior_,
        total_samples=model.total_samples,
        sub_sampling=True,
    )
    expected_sub = model._approx_bound(X, doc_topic, sub_sampling=True)
    assert np.allclose(observed_sub, expected_sub)


def test_lda_bound_contracts_reject_invalid_inputs() -> None:
    with pytest.raises(ViolationError):
        lda_dirichlet_loglikelihood(0.0, np.ones((2, 2)), np.ones((2, 2)), 2)

    with pytest.raises(ViolationError):
        lda_document_log_probability_bound(
            np.array([[1.0, -1.0]]),
            np.ones((1, 2)),
            np.ones((2, 2)),
        )

    with pytest.raises(ViolationError):
        lda_apply_subsampling_ratio(1.0, total_samples=1.0, current_samples=0)

